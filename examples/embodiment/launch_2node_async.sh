#!/bin/bash
# ==============================================================================
# Multi-Node Ray Cluster Startup
#
# Usage:
#   Head node:   RLINF_NODE_RANK=0 HEAD_IP=192.168.251.176 bash launch_2node_async.sh
#   Worker node: RLINF_NODE_RANK=1 HEAD_IP=192.168.251.176 bash launch_2node_async.sh
# ==============================================================================

set -e

RAY_PORT="${RAY_PORT:-6379}"

if [ -z "$RLINF_NODE_RANK" ]; then
    echo "Error: RLINF_NODE_RANK must be set (0=head, 1,2,...=worker)"
    exit 1
fi
if [ -z "$HEAD_IP" ]; then
    echo "Error: HEAD_IP must be set to head node's NIC IP"
    exit 1
fi

export RLINF_NODE_RANK

ray stop 2>/dev/null || true
sleep 2

if [ "$RLINF_NODE_RANK" -eq 0 ]; then
    echo "=== Ray HEAD (node $RLINF_NODE_RANK) on $HEAD_IP:$RAY_PORT ==="
    ray start --head --port=$RAY_PORT --node-ip-address=$HEAD_IP
else
    echo "=== Ray WORKER (node $RLINF_NODE_RANK) -> $HEAD_IP:$RAY_PORT ==="
    ray start --address="$HEAD_IP:$RAY_PORT"
fi

ray status
echo "=== RLINF_NODE_RANK=$RLINF_NODE_RANK ready ==="
