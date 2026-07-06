#!/usr/bin/env bash
# One-command build + push + roll for the console image.
#
# Bakes BOTH the lerobot base tag and this console's commit into the image, so the UI's
# "更新说明" version banner is always correct — no manual tag juggling:
#   - lerobot commit  = the tag of BASE_IMAGE (shown from LEROBOT_IMAGE env).
#   - console commit  = git HEAD, written to static/version.txt AND passed as --build-arg
#                       CONSOLE_COMMIT (belt-and-suspenders; works for clone/CI builds too).
#
# Usage:
#   KUBECONFIG=~/Downloads/kube.conf ./scripts/redeploy.sh          # build + push + roll pod
#   DEPLOY=0 ./scripts/redeploy.sh                                  # build + push only
#   BASE_IMAGE=.../lerobot:<tag> ./scripts/redeploy.sh             # override the lerobot base
#
# Prereqs: docker buildx logged in to the registry; kubectl reachable (for DEPLOY=1).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REGISTRY="${REGISTRY:-iaas-us-cn-beijing.cr.volces.com/physicalai}"
# Default the lerobot base to whatever the Dockerfile currently pins.
BASE_IMAGE="${BASE_IMAGE:-$(grep -oE 'iaas-[^ ]*/lerobot:[0-9a-f]+' Dockerfile | head -1)}"
COMMIT="$(git rev-parse HEAD)"
IMAGE="$REGISTRY/lerobot-agent-console:$COMMIT"

echo "base image : $BASE_IMAGE"
echo "console img: $IMAGE"
[ -n "$BASE_IMAGE" ] || { echo "ERROR: BASE_IMAGE not set and none found in Dockerfile"; exit 1; }

# Make the console commit available to the running server two ways (file + build-arg).
git rev-parse HEAD > static/version.txt

docker buildx build \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg CONSOLE_COMMIT="$COMMIT" \
  --output "type=image,name=$IMAGE,push=true,compression=gzip,oci-mediatypes=true" \
  .

if [ "${DEPLOY:-1}" = "1" ]; then
  echo "rolling deploy/lerobot-console → $COMMIT"
  kubectl set image deploy/lerobot-console "console=$IMAGE"
  kubectl rollout status deploy/lerobot-console
fi
echo "done: $IMAGE"
