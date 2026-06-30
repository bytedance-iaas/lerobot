#!/usr/bin/env bash
# Deploy the self-hosted LiveKit SFU so the cloud can reach a robot on your Mac.
# Solves the chicken-and-egg: create the CLB first, read its public IP, then start
# LiveKit with --node-ip=<that IP> (so ICE candidates point clients at the CLB).
#
#   KUBECONFIG=~/Downloads/kube.conf ./scripts/deploy-livekit.sh
#
# After this, `kubectl delete pod -l app=livekit` recreates with the same spec
# (node-ip is already baked into the Deployment). Only a fresh CLB (new IP) needs
# a re-run. The console pod (which runs the controller) is deployed separately via
# the k8s/ manifests; the controller's web panel is shown in the console viewer
# (via /proxy), so no separate web CLB is needed.
set -euo pipefail
: "${KUBECONFIG:=$HOME/Downloads/kube.conf}"; export KUBECONFIG
NS="${NS:-default}"
DIR="$(cd "$(dirname "$0")/../k8s/livekit" && pwd)"

wait_lb_ip() {  # $1 = service name -> prints public IP (or empty after timeout)
  for _ in $(seq 1 80); do
    ip=$(kubectl get svc "$1" -n "$NS" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
    [ -n "$ip" ] && { echo "$ip"; return; }
    sleep 3
  done
}

echo "==> [1/3] LiveKit config + public CLB"
kubectl apply -f "$DIR/configmap.yaml"
kubectl apply -f "$DIR/service-clb.yaml"

echo "==> [2/3] waiting for livekit-clb public IP ..."
LK_IP=$(wait_lb_ip livekit-clb)
[ -n "$LK_IP" ] || { echo "ERROR: livekit-clb got no public IP (check the subnet-id annotation)"; exit 1; }
echo "    livekit public IP = $LK_IP"

echo "==> [3/3] LiveKit server with --node-ip=$LK_IP"
sed "s/__NODE_IP__/$LK_IP/" "$DIR/deployment.yaml" | kubectl apply -f -

cat <<EOF

==================== LiveKit READY ====================
Mac dials OUT to:   ws://$LK_IP:7880   (tcp 7881 / udp 7882 on the same IP)
LiveKit key/secret: devkey / lerobotlivekitsecret0123456789abcd  (CHANGE for prod)

NEXT — on the home Mac (real SO-100 + cameras):
  .venv/bin/python -m lerobot.robots.webrtc_proxy.mac_daemon \\
    --transport livekit --session so100 \\
    --livekit-url ws://$LK_IP:7880 \\
    --livekit-api-key devkey --livekit-api-secret lerobotlivekitsecret0123456789abcd \\
    --robot.type=so100_follower --robot.port=/dev/tty.usbmodemXXXX \\
    --robot.id=my_follower --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480} }"

THEN — in the LeRobot Agent Console terminal (same pod as the controller):
  cd /lerobot && setsid nohup ./.venv/bin/python \\
    examples/webrtc_remote_so100/cloud_teleop_so100.py \\
    --mode web --transport livekit --session so100 --cameras "front,wrist" --web-port 8088 \\
    --livekit-url ws://livekit-clb:7880 \\
    --livekit-api-key devkey --livekit-api-secret lerobotlivekitsecret0123456789abcd \\
    > /tmp/teleop.log 2>&1 < /dev/null & echo started

Then in the console's left viewer: "+ 打开" -> 8088  (the panel is proxied, no extra CLB).
=======================================================
EOF
