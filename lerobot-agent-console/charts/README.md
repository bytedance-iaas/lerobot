## Installing

The real install path is the **VKE console → 创建 Helm 应用** page: pick the chart, edit
`values.yaml` in the text box, name the release, deploy. There is no shell in that flow, so the
charts create everything they need — with one deliberate exception: credentials. Those come from
a Secret you make first, on the console's own **密钥 (Secret)** page, which needs no shell either.

`values.yaml` therefore ships **installable-anywhere defaults**, not our deployment:
- `image.tag` is empty and required. The chart deliberately does not track the image: the
  pipeline tags every build with its commit sha and publishes no `latest`, so pinning one here
  would mean republishing the chart for every build. Chart version and image version are
  different things. Pass `--set image.tag=<sha>` (or set it in the values box), and check the
  tag exists first — `oras repo tags <repo>` — because a sha that was never built gives
  ImagePullBackOff.
- `auth.existingSecret` names a Secret you create beforehand; the install FAILS until it is set.
  The chart will not take credentials through values: Helm keeps values verbatim in the release
  history, so `helm get values` reads them back — on every past revision, which means rotating a
  password does not remove the old one. Create it on the VKE console's 密钥 page (no shell
  needed) or with `kubectl create secret generic <name> --from-literal=username=… --from-literal=password=…`,
  in the SAME namespace as the release. Key names default to `username`/`password`; point
  `auth.usernameKey` / `auth.passwordKey` at whatever your Secret already uses.
- `nodeSelector: {}` — the GPU request schedules the pod. A hostname from our cluster would
  leave it Pending forever on yours.
- `apig.enabled: false` — the console works in-cluster without it. Turn it on when you know which
  gateway to use: `create: false` + `existingId` adopts one from the APIG console, `create: true`
  + `subnetIds` provisions a new one. ⚠️ A provisioned gateway is deleted by `helm uninstall`,
  taking its `*.volceapi.com` domain with it.

  **`create: true` is a two-step bootstrap.** The Ingress binds to a gateway through a
  `loadbalancer-id` annotation, and that id does not exist until the gateway has been
  provisioned — so the first render cannot carry it. Verified on a real install: without it the
  Ingress never gets an address and APIG lists no service for the gateway; setting the id made
  both appear within a minute. After the first install:

  ```bash
  kubectl get apiginstance <release>-apig -o jsonpath='{.status.id}'
  ```
  put that in `apig.existingId` (leave `create: true`) and upgrade.

  The gateway's public `*.volceapi.com` name is assigned by APIG and is **not exposed anywhere in
  Kubernetes** — not in the CRD status, not on the Ingress. Read it from the APIG console. For
  anything long-lived, bind your own domain there and CNAME it, so the URL survives the
  gateway.
- `livekit.nodeIp` may be left empty: an init container waits for this chart's own CLB to be
  assigned a public IP and passes it to the server. Set it explicitly to skip the wait — the
  init container and its RBAC then disappear from the release.

Cluster-specific values you will have to change regardless: `persistence.storageClass`,
`livekit.service.subnetId`, and `image.repository` if you mirror the image.

### LiveKit: where nodeIp comes from

`nodeIp` is the address LiveKit hands clients as their ICE candidate. It must be the **public IP
of this release's CLB**, and the chart is what creates that CLB — so the value does not exist
until after the first install. An init container closes that gap: it waits for the Service to be
assigned an address (up to 10 minutes, logging each attempt) and writes it out for the server.
Setting `nodeIp` explicitly skips it, and the ServiceAccount/Role go with it.

⚠️ **It cannot be discovered by STUN, and getting it wrong fails misleadingly**: signalling
connects, the call looks alive, media never flows. LiveKit's `use_external_ip` finds the address
by STUN — that is the pod's *egress* IP, which on this cluster measured 124.174.58.65 while the
CLB's *ingress* IP was 115.190.7.216. Two different addresses; nothing listens on the egress one.

If the address matters long-term, allocate an EIP yourself and bind the CLB to it: `nodeIp` is
then known before the first install and survives re-creating the release.

`helm uninstall` deliberately leaves the CLB behind (`helm.sh/resource-policy: keep`), because
deleting it does not give the public IP back and every client has that IP in its `--livekit-url`.
For a throwaway install that protection is only a billed leftover and an EIP out of quota, so set
`service.keepOnUninstall: false` and uninstall cleans up after itself.

### Deploying with the CLI instead

There are no per-environment values files in this repo on purpose — an environment's file wants
to hold its password, and that is not a thing to commit (nor, since the `existingSecret` change,
a thing the charts will read). Generate one at deploy time:

```bash
cat charts/lerobot-agent-console/values.yaml > /tmp/values-activate.yaml   # start from the defaults
# then append the environment's overrides, e.g. auth.existingSecret, image.tag, apig.*
helm upgrade --install <release> charts/lerobot-agent-console \
  -f /tmp/values-activate.yaml --reset-values
```

`values-activate.yaml` is gitignored anywhere under `charts/`. Always pass `--reset-values`:
without it `helm upgrade` reuses the previous release's values, and a stale `image.tag` override
silently outranks the one in the chart.

# Helm charts

Replaces the hand-maintained manifests in `k8s/`. Those were gitignored, so they only ever
existed on one laptop, and the two console StatefulSets were kept in sync by editing both —
they turned out to be identical apart from the name, which is what one chart plus two values
files is for.

| chart | deploys | notes |
|---|---|---|
| `lerobot-agent-console` | console StatefulSet + headless Service | dev and test are the **same chart**, differing only by `nameOverride` |
| `livekit` | LiveKit SFU: Deployment + ConfigMap + its own public CLB | **shared by both consoles** — see below |

## Install / upgrade

```bash
export KUBECONFIG=~/Downloads/kube.conf

# dev
helm upgrade --install lerobot-agent-console charts/lerobot-agent-console

# test — same chart, one value different
helm upgrade --install lerobot-agent-console-test charts/lerobot-agent-console \
  -f charts/lerobot-agent-console/values-test.yaml

# livekit — nodeIp is REQUIRED (the chart refuses to render without it: a wrong value makes
# ICE hand clients an unroutable address and media never connects)
helm upgrade --install livekit charts/livekit \
  --set nodeIp=$(kubectl get svc livekit-clb -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
```

Ship a new image without touching the chart:

```bash
helm upgrade lerobot-agent-console charts/lerobot-agent-console \
  --set image.tag=$(git rev-parse HEAD) --reuse-values
```

**Resource names come from `nameOverride`, not the release name.** That value decides the PVC
name (`hermes-home-<name>-0`) and the pod DNS the multi-node training dials, so it is pinned in
values rather than inherited — installing under a different release name still produces the
right objects.

## Inspecting what is actually deployed

```bash
helm list                                    # releases, revision, status
helm get manifest lerobot-agent-console      # the YAML really submitted — the useful one
helm get values  lerobot-agent-console       # values in effect
helm history     lerobot-agent-console       # revisions; input to `helm rollback <name> <rev>`

# has anyone edited around Helm?
helm get manifest lerobot-agent-console | kubectl diff -f -
```

Helm keeps no local state: each release is a Secret in the namespace
(`kubectl get secret -l owner=helm`), so any machine with the same kubeconfig sees the same
releases. `STATUS: deployed` means the Helm operation succeeded — it says nothing about whether
the pods are healthy.

## Why LiveKit is a separate chart

The console does depend on it — the in-pod controller dials the SFU for teleoperation — so a
subchart is tempting. Two things argue against:

- **`nodeIp` is chicken-and-egg.** LiveKit starts with `--node-ip = the CLB's public IP`, and
  that IP does not exist until the Service has been provisioned. Bundled, one `helm install`
  cannot finish the job.
- **The CLB outlives any console.** It is billed, slow to provision, and its IP is baked into
  every client's `--livekit-url`. Bundling means `helm uninstall` of a console takes the SFU's
  address with it.

**One SFU serves both consoles.** LiveKit isolates by room and the teleop commands already pass
one (`--session so100`) — point dev at `so100` and test at `so100-test`. A second SFU only earns
its CLB when you need to change LiveKit itself without disturbing what dev is running.
`livekit-isaac` is already that shape: a second SFU paired with no console at all (it serves
isaaclab), which is the other reason this chart has to stand alone.

The CLB Service carries `helm.sh/resource-policy: keep`, so even `helm uninstall livekit` leaves
it — and the public IP — in place. Remove it deliberately: `kubectl delete svc <name>-clb`.

## Two immutable fields that will bite

**`volumeClaimTemplates` cannot change on an existing StatefulSet.** `persistence.size` is
`1Ti`, matching the real disks. If a running StatefulSet's template says something else, the
API server rejects the upgrade outright — recreate the object with
`kubectl delete sts <name> --cascade=orphan` (pods keep running, PVCs survive) and install
again. To grow a disk later, edit the PVC **and** this value, or the next recreate silently
goes back down.

**`spec.selector` cannot change either.** The consoles use a bare `app: <name>`, so
`console.selectorLabels` emits exactly that. Adding the conventional `app.kubernetes.io/*` or
`helm.sh/chart` labels there breaks every upgrade — extra labels belong on metadata.

**Check with an apply dry-run, not `kubectl diff`.** `kubectl diff` writes an immutable-field
rejection to *stderr* and exits 2, so a pipeline that discards stderr reports "no differences"
for a change the cluster will not accept:

```bash
helm template lerobot-agent-console charts/lerobot-agent-console | kubectl apply --dry-run=server -f -
```

## Secrets are not in the charts

Both charts reference Secrets by name and never create them:

```bash
kubectl create secret generic <console-secret> --from-literal=username=<u> --from-literal=password=<p>
kubectl create secret generic <livekit-secret> --from-literal=keys='<api_key>: <api_secret>'
```

Neither chart has a credential fallback, on purpose: one hands out a root shell on a GPU node,
the other signs access tokens for an SFU on a public CLB. Point `auth.existingSecret` at the
Secret's name; it must be in the same namespace as the release. Rendering fails without it, which
is a better failure than starting.

```bash
helm upgrade --install <release> charts/lerobot-agent-console \
  --set image.tag=<sha> --set auth.existingSecret=lerobot-console-auth
```

`auth.usernameKey` / `auth.passwordKey` exist for adopting a Secret that already uses other key
names, without rebuilding it around a live password. Ours needed that once — `lerobot-console-auth`
predates the chart and stored the name under `user` — but it now carries `username` as well, so
the defaults apply and nothing overrides them. The old `user` key is still there because the
running release references it; drop it once every release is on a chart ≥ 0.4.

## Publishing to the OCI registry

```bash
export HELM_REGISTRY_HOST=iaas-us-cn-beijing.cr.volces.com
export HELM_REGISTRY_NAMESPACE=physicalai
export HELM_REGISTRY_USERNAME=<account>@<account-id> HELM_REGISTRY_PASSWORD=<password>

DRY_RUN=1 bash scripts/publish_charts.sh          # package + gate only, touches no registry
bash scripts/publish_charts.sh                    # push every chart
bash scripts/publish_charts.sh livekit            # or just one
```

Adapted from RLinf's `docker/publish_charts.sh`, so the same script works on a CI runner with no
helm, no root and no package manager: it unpacks a static helm into a temp dir, and logs in
through a throwaway `--registry-config` so the robot account is not left behind for the next job.

Before packaging, each chart is rendered with placeholder values. That gate is `helm template`,
not `helm lint`: on helm v4.2.3 a template that calls `fail` — which is how both charts refuse a
missing `auth.existingSecret` — lints as `level=INFO` and still reports "0 chart(s) failed",
exit 0. Placeholder values live in a `case` block in the script; a chart added without an entry
there stops the run rather than being published unrendered.

The push refuses a version already in the registry (`ALLOW_OVERWRITE=1` to override). That is the
same rule `scripts/check-chart-version.sh` enforces against git, asked of the registry instead, so
it still holds in a shallow CI clone where no base ref resolves.

The chart is named after the product, so it lands in the **same repo as the image**
(`physicalai/lerobot-agent-console`). They coexist: image tags are 40-char commit SHAs, chart
tags are semver. Bump `version:` in `Chart.yaml` for any template change — pushing over an
existing version is how you get two different charts answering to one number.

Credentials come from the macOS keychain (`credsStore`), so `auths` in the config JSON is empty
even when you are logged in. A read probe returning **404 rather than 401** means auth worked.

## Not converted

`k8s/apig-ingress*.yaml` and `k8s/apig-instance-test.yaml` stay raw. They create Volcengine
`APIGInstance` CRDs that own **provisioned gateways with public domains**; a stray
`helm uninstall` would delete the gateway and the domain with it. They change roughly never.
