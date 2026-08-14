#!/usr/bin/env bash
# Deploy the self-hosted LiveKit SFU on its OWN dedicated CLB / public IP. LiveKit no
# longer shares the console's CLB — the console moved to APIG (see k8s/apig-ingress.yaml).
# This script:
#   - applies k8s/livekit/service-clb.yaml, which OWNS a CLB (created from its subnet-id),
#     with 7880/7881/7882 listeners
#   - waits for that livekit CLB's public IP
#   - runs LiveKit with --node-ip = that IP (ICE points clients at the livekit CLB)
#
#   KUBECONFIG=~/Downloads/kube.conf ./scripts/deploy-livekit.sh
#
# Self-contained: it no longer depends on the console CLB. `kubectl delete pod -l app=livekit`
# recreates livekit with the same spec (node-ip baked in). Re-run only if the livekit CLB IP
# changes. Teardown: `kubectl delete -f k8s/livekit/service-clb.yaml` deletes the livekit CLB.
set -euo pipefail
: "${KUBECONFIG:=$HOME/Downloads/kube.conf}"; export KUBECONFIG
NS="${NS:-default}"
DIR="$(cd "$(dirname "$0")/../k8s/livekit" && pwd)"
LK_SVC="${LK_SVC:-livekit-clb}"

# Credentials are env-injected from the livekit-auth Secret, and the ConfigMap carries no
# fallback (see configmap.yaml). Check it up front: otherwise the pods sit in
# CreateContainerConfigError AFTER the CLB has been provisioned, which reads as "livekit is
# broken" instead of "you never created the Secret".
if ! kubectl get secret livekit-auth -n "$NS" >/dev/null 2>&1; then
  cat >&2 <<'MSG'
ERROR: Secret `livekit-auth` is missing — LiveKit has no API key/secret to run with.
This SFU is published on a public CLB, so it deliberately has NO usable default: a key
committed to git would mean anyone who can read the repo can join your rooms.

Generate a pair and create the Secret (do not commit the value anywhere):

  kubectl create secret generic livekit-auth \
    --from-literal=keys='<api_key>: <api_secret>'

`livekit-server generate-keys` produces a pair; any 32+ char random secret works.
Every client (robot daemon, in-pod controller) must pass the SAME pair as
--livekit-api-key / --livekit-api-secret.
MSG
  exit 1
fi

echo "==> LiveKit config + its OWN CLB (7880/7881/7882 on a dedicated public IP)"
kubectl apply -f "$DIR/configmap.yaml"
kubectl apply -f "$DIR/service-clb.yaml"          # owns the CLB via subnet-id (no more __SHARED_CLB_ID__)

echo "==> waiting for the livekit CLB public IP ..."
CLB_IP=""
for _ in $(seq 1 80); do
  CLB_IP=$(kubectl get svc "$LK_SVC" -n "$NS" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
  [ -n "$CLB_IP" ] && break
  sleep 3
done
[ -n "$CLB_IP" ] || { echo "ERROR: livekit CLB ($LK_SVC) got no public IP — check the subnet-id in service-clb.yaml"; exit 1; }
echo "    livekit CLB ip=$CLB_IP"

echo "==> LiveKit server with --node-ip=$CLB_IP (the livekit CLB IP)"
sed "s/__NODE_IP__/$CLB_IP/" "$DIR/deployment.yaml" | kubectl apply -f -

cat <<EOF

==================== LiveKit READY (dedicated CLB) ====================
LiveKit has its OWN CLB / public IP: $CLB_IP   (console is separate, on APIG)
Mac dials OUT to:   ws://$CLB_IP:7880   (tcp 7881 / udp 7882 on the same IP)
LiveKit key/secret: from the livekit-auth Secret (never printed here — read it with
                    kubectl get secret livekit-auth -o jsonpath='{.data.keys}' | base64 -d)

NEXT — on the home Mac (real SO-100 + cameras):
  .venv/bin/python -m lerobot.robots.webrtc_proxy.mac_daemon \\
    --transport livekit --session so100 \\
    --livekit-url ws://$CLB_IP:7880 \\
    --livekit-api-key <api_key> --livekit-api-secret <api_secret> \\
    --robot.type=so100_follower --robot.port=/dev/tty.usbmodemXXXX \\
    --robot.id=my_follower --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480} }"

THEN — in the console terminal (in-cluster it dials the internal service name):
  cd /lerobot && setsid nohup ./.venv/bin/python \\
    examples/webrtc_remote_so100/cloud_teleop_so100.py \\
    --mode web --transport livekit --session so100 --cameras "front,wrist" --web-port 8088 \\
    --livekit-url ws://livekit-clb:7880 \\
    --livekit-api-key <api_key> --livekit-api-secret <api_secret> \\
    > /tmp/teleop.log 2>&1 < /dev/null & echo started
======================================================================
EOF
