#!/usr/bin/env bash
# Package the console Helm charts and push them to the Volcengine OCI registry.
#
#   bash scripts/publish_charts.sh                    # package, log in, push all charts
#   bash scripts/publish_charts.sh livekit            # just one chart
#   DRY_RUN=1 bash scripts/publish_charts.sh          # package + gate only, no registry contact
#
# Everything deployment-specific comes from the environment, so this file holds no
# registry coordinates and no credentials:
#   HELM_REGISTRY_HOST       registry hostname, e.g. example.cr.volces.com
#   HELM_REGISTRY_NAMESPACE  namespace the charts are pushed under
#   HELM_REGISTRY_USERNAME   robot account      (not needed for DRY_RUN)
#   HELM_REGISTRY_PASSWORD   its password       (not needed for DRY_RUN)
#
# Optional:
#   HELM_VERSION             helm to fetch when the runner has none (default: latest)
#   ALLOW_OVERWRITE=1        push even if that chart version is already in the registry
#
# Nothing here needs root or a package manager. The CI image runs as nobody on a
# release old enough that its package sources are gone, so the only things
# assumed present are bash, curl and tar.
#
# Adapted from bytedance-iaas/RLinf docker/publish_charts.sh. Two differences, both
# because our charts refuse to render without credentials (see charts/README.md):
#
#   * `helm lint` is NOT a render gate. Measured on helm v4.2.3: a template that calls
#     `fail` lints as `level=INFO msg="funcMap fail"` and still reports "0 chart(s)
#     failed", exit 0. So each chart is rendered with `helm template` under placeholder
#     values, which does exit 1. lint stays for the chart-metadata checks it does do.
#   * The required values differ per chart, so they live in a case block below rather
#     than on one shared lint line. A chart added without an entry there fails loudly.

set -euo pipefail

# Checked before anything else: a missing coordinate is a configuration error, and
# failing here costs nothing, whereas failing after the helm download wastes the
# whole setup. DRY_RUN needs these too, since it reports the push target.
: "${HELM_REGISTRY_HOST:?Set HELM_REGISTRY_HOST, the registry hostname (e.g. example.cr.volces.com).}"
: "${HELM_REGISTRY_NAMESPACE:?Set HELM_REGISTRY_NAMESPACE, the namespace the charts are pushed under.}"
registry="${HELM_REGISTRY_HOST}"
namespace="${HELM_REGISTRY_NAMESPACE}"

# Resolve from the script's own location so the working directory does not matter.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# Placeholder values that satisfy each chart's required-value gates. These exist only
# to make the templates render; nothing here is ever deployed. Note livekit's nodeIp
# must parse as an address — TEST-NET-3 (RFC 5737), reserved for documentation.
render_values() {
  case "$1" in
    lerobot-agent-console)
      echo "--set image.tag=ci-lint --set auth.existingSecret=ci-lint" ;;
    livekit)
      echo "--set nodeIp=203.0.113.1 --set service.subnetId=ci-lint --set auth.existingSecret=ci-lint" ;;
    *)
      echo "error: chart '$1' has no render values in publish_charts.sh — add a case for it" >&2
      return 1 ;;
  esac
}

charts=("$@")
if [[ ${#charts[@]} -eq 0 ]]; then
  for dir in charts/*/; do
    [[ -f "${dir}Chart.yaml" ]] && charts+=("$(basename "${dir}")")
  done
fi
[[ ${#charts[@]} -gt 0 ]] || { echo "error: no charts found under charts/" >&2; exit 1; }

echo "=== environment ==="
echo "user:   $(id -un 2>/dev/null || echo unknown) (uid $(id -u))"
echo "os:     $(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}" || uname -s) $(uname -m)"
echo "helm:   $(command -v helm >/dev/null 2>&1 && helm version --short 2>&1 || echo MISSING)"
echo "curl:   $(command -v curl || echo MISSING)"
echo "tar:    $(command -v tar || echo MISSING)"
echo "charts: ${charts[*]}"
echo "==================="

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

# helm ships a statically linked binary, so unpacking it into the work directory
# needs no privileges — the only option available here, since the runner is
# nobody, sudo cannot elevate, and there is no usable package source.
if ! command -v helm >/dev/null 2>&1; then
  case "$(uname -m)" in
    x86_64 | amd64) helm_arch=amd64 ;;
    aarch64 | arm64) helm_arch=arm64 ;;
    *) echo "error: unsupported architecture $(uname -m)" >&2; exit 1 ;;
  esac
  helm_os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  helm_version="${HELM_VERSION:-$(curl -fsSL --max-time 30 https://get.helm.sh/helm-latest-version 2>/dev/null || echo v4.2.3)}"
  helm_url="https://get.helm.sh/helm-${helm_version}-${helm_os}-${helm_arch}.tar.gz"

  echo "helm missing, fetching ${helm_url}"
  mkdir -p "${work}/helm"
  curl -fsSL --max-time 300 "${helm_url}" | tar -xz -C "${work}/helm" ||
    { echo "error: could not download helm from ${helm_url}" >&2; exit 1; }
  export PATH="${work}/helm/${helm_os}-${helm_arch}:${PATH}"

  command -v helm >/dev/null 2>&1 ||
    { echo "error: helm still not on PATH after unpacking" >&2; exit 1; }
  echo "helm ready: $(helm version --short 2>&1)"
fi

# Name and version come from Chart.yaml, which is also where helm reads them, so each
# package is simply whatever lands in its own otherwise-empty directory. That keeps a
# chart's identity in one place and this script out of the business of parsing YAML.
declare -a packages=()
for chart in "${charts[@]}"; do
  dir="charts/${chart}"
  [[ -f "${dir}/Chart.yaml" ]] || { echo "error: no chart at ${dir}" >&2; exit 1; }
  echo "=== ${chart} ==="

  read -r -a values <<< "$(render_values "${chart}")"
  helm lint "${dir}" "${values[@]}"
  # The actual gate — see the header note on lint and `fail`.
  helm template gate "${dir}" "${values[@]}" >/dev/null

  mkdir -p "${work}/pkg/${chart}"
  helm package "${dir}" --destination "${work}/pkg/${chart}"
  # Name and version are read from Chart.yaml rather than split out of the filename:
  # `<name>-<version>.tgz` is ambiguous once a version carries a prerelease tag, and
  # livekit-0.5.0-rc.1.tgz would split as name=livekit-0.5.0 version=rc.1 — coordinates
  # that exist nowhere, so the already-published check below would probe a miss and pass.
  cname="$(grep -E '^name:' "${dir}/Chart.yaml" | head -1 | awk '{print $2}')"
  cver="$(grep -E '^version:' "${dir}/Chart.yaml" | head -1 | awk '{print $2}')"
  packages+=("${cname}|${cver}|$(echo "${work}/pkg/${chart}"/*.tgz)")
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for entry in "${packages[@]}"; do
    IFS='|' read -r name version package <<< "${entry}"
    echo "dry run: would push ${name} ${version} (${package##*/}) to oci://${registry}/${namespace}"
  done
  exit 0
fi

: "${HELM_REGISTRY_USERNAME:?Set HELM_REGISTRY_USERNAME (registry robot account).}"
: "${HELM_REGISTRY_PASSWORD:?Set HELM_REGISTRY_PASSWORD.}"

# Keep the login in a throwaway config: the default ~/.config/helm would leave
# the robot credentials on a reused CI runner for the next job to find.
config="${work}/registry.json"
printf '%s' "${HELM_REGISTRY_PASSWORD}" |
  helm registry login "${registry}" --username "${HELM_REGISTRY_USERNAME}" \
    --password-stdin --registry-config "${config}"

for entry in "${packages[@]}"; do
  IFS='|' read -r name version package <<< "${entry}"

  # helm push overwrites an existing tag without complaint, which would leave two
  # different charts answering to one version — the same failure scripts/check-chart-version.sh
  # guards against in git. This asks the registry instead, so it still holds on a shallow
  # clone where that script cannot resolve a base ref.
  if [[ "${ALLOW_OVERWRITE:-0}" != "1" ]] &&
     helm show chart "oci://${registry}/${namespace}/${name}" --version "${version}" \
       --registry-config "${config}" >/dev/null 2>&1; then
    echo "error: ${name} ${version} is already in the registry." >&2
    echo "       Bump version: in charts/${name}/Chart.yaml, or set ALLOW_OVERWRITE=1 if you" >&2
    echo "       really mean to replace it." >&2
    exit 1
  fi

  # helm push reports the pushed reference and its digest on success.
  helm push "${package}" "oci://${registry}/${namespace}" --registry-config "${config}"
done
