# 2-Node Pi0.5 DSRL Benchmark Guide (RTX PRO 5000 Blackwell)

## Prerequisites

**Hardware**:
- 2 nodes with NVIDIA GPUs (tested on RTX PRO 5000 Blackwell, sm_120), IB 400Gbps interconnect
- Reference setup: Node 0 (8 GPU) + Node 1 (4 GPU)

**Software**:

| Component | Source | Local path (reference) |
|-----------|--------|----------------------|
| Container image | DockerHub:`chenchaox72877/rlinf:0.2-maniskill_libero-blackwell` | `/root/chenchaox/rlinf_pub/env/images/rlinf0.2-maniskill_libero_blackwell_v2.sqsh` |
| Model checkpoint(pi0.5) | HuggingFace:`https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT` | `/workspace/rlinf_pub/models/RLinf-Pi05-LIBERO-SFT/` |
| RLinf repo | GitHub: [`chenchaoxu7575/RLinf`](https://github.com/chenchaoxu7575/RLinf) (branch: `pi05-turtle-2node-async`) | `/root/chenchaox/rlinf_pub/RLinf` |

Container image requirements: CUDA 12.8, torch 2.7.1+cu128, NCCL 2.28.9, flash-attn 2.7.4.post1, nccl-rdma-sharp-plugins (rebuilt for NCCL 2.28.9).

**Setup**:
- Enroot container runtime installed on both nodes
- Code and model accessible from both nodes (shared filesystem or synced via `rsync`)

---

## Step 1: Prepare Workspace (both nodes)

> **Note**: The two nodes do NOT share a filesystem. All setup steps below must be performed on BOTH nodes independently. Use `rsync` to keep code in sync after changes.

```bash
export WORKDIR={YOUR_WORKDIR}   # e.g. /root/chenchaox

# Create workspace directory
mkdir -p $WORKDIR/rlinf_pub
cd $WORKDIR/rlinf_pub

# Clone RLinf repo
git clone https://github.com/chenchaoxu7575/RLinf.git RLinf
cd RLinf && git checkout pi05-turtle-2node-async && cd ..

# Download model checkpoint
mkdir -p models
# Option A: from HuggingFace
# huggingface-cli download <MODEL_ID> --local-dir models/RLinf-Pi05-LIBERO-SFT
# Option B: from existing machine
# rsync -avz root@<OTHER_NODE>:$WORKDIR/rlinf_pub/models/ models/

# Download container image and import to Enroot
# Option A: from DockerHub
# enroot import 'docker://<IMAGE_URL>' -o env/images/rlinf-blackwell.sqsh
# Option B: from existing machine
# rsync -avz root@<OTHER_NODE>:$WORKDIR/rlinf_pub/env/images/*.sqsh env/images/
enroot create --name rlinf-blackwell env/images/rlinf-blackwell.sqsh
```

After initial setup, sync code changes between nodes:
```bash
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='logs/' --exclude='models/' \
  $WORKDIR/rlinf_pub/ root@<OTHER_NODE_IP>:$WORKDIR/rlinf_pub/
```

## Step 2: Host Setup (both nodes)

```bash
# Start IB Subnet Manager (head node only)
sudo opensm &

# Verify IB is active (both nodes)
ibstat | grep -A5 "Port 1"
# Expect: State: Active, Physical state: LinkUp, Rate: 400
```

## Step 3: Start Containers (both nodes)

```bash
enroot start --rw \
  --mount $WORKDIR/rlinf_pub:/workspace/rlinf_pub \
  --mount /dev/infiniband:/dev/infiniband \
  rlinf-blackwell
```

## Step 4: Verify Environment (inside containers)

```bash
source switch_env openpi

# Versions
python -c "import torch; print(torch.__version__)"           # 2.7.1+cu128
python -c "import flash_attn; print(flash_attn.__version__)"  # 2.7.4.post1
pip show nvidia-nccl-cu12 | grep Version                      # 2.28.9

# IB device
ibv_devinfo | head -10

# Cross-node connectivity
ping -c 3 <other_node_ip>

# Model checkpoint
ls /workspace/rlinf_pub/models/RLinf-Pi05-LIBERO-SFT/model.safetensors
```

## Step 5: Start Ray Cluster

Determine the ethernet IP of each node (must be routable between nodes):
```bash
# Run on each node to find IP
ip addr show | grep "inet " | grep -v 127.0.0.1
```

**Node 0 (head):**
```bash
export HEAD_IP=<NODE0_ETH_IP>   # e.g. 192.168.1.100
ulimit -n 65536
export RLINF_NODE_RANK=0
ray start --head --port=6379 --node-ip-address=$HEAD_IP
```

**Node 1 (worker):**
```bash
export HEAD_IP=<NODE0_ETH_IP>   # same IP as above
ulimit -n 65536
export RLINF_NODE_RANK=1
ray start --address="$HEAD_IP:6379"
```

Verify: `ray status` shows 2 nodes.

## Step 6: Run Benchmark

Run experiments from the **head node** (Node 0) container. Ray will automatically dispatch workers to both nodes.

### A/B Comparison: GLOO vs NCCL Weight Sync

**Experiment A — GLOO baseline** (weight sync via TCP/10GbE):
```bash
cd /workspace/rlinf_pub/RLinf
bash examples/embodiment/run_async.sh realworld_dummy_turtle2_dsrl_pi05_2node_1rank_rtx5kpro_async_gloo
```

**Experiment B — NCCL/IB optimized** (weight sync via NCCL/IB 400G + GDR):
```bash
cd /workspace/rlinf_pub/RLinf
bash examples/embodiment/run_async.sh realworld_dummy_turtle2_dsrl_pi05_2node_1rank_rtx5kpro_async_nccl
```

Both configs use identical model/algorithm settings. The only difference is communication backend:

| | GLOO (Exp A) | NCCL (Exp B) |
|---|---|---|
| Config | `_2node_1rank_*_gloo` | `_2node_1rank_*_nccl` |
| `RLINF_FORCE_ACCEL_CCL` | not set | `1` |
| `NCCL_NET_GDR_LEVEL` | not set | `SYS` |
| `NCCL_IB_HCA` | not set | `mlx5_0` |
| Weight sync transport | GLOO/TCP over 10GbE | NCCL/IB 400G + GDRDMA |
| Expected weight sync time | ~17.5s | ~242ms |

## Step 7: Collect and Compare Results

### Quick comparison from logs

After both experiments complete, run on head node (inside container):

```bash
# Set log dirs (auto-detect latest runs)
GLOO_LOG=$(ls -td /workspace/rlinf_pub/RLinf/logs/*gloo* | head -1)
NCCL_LOG=$(ls -td /workspace/rlinf_pub/RLinf/logs/*nccl*1rank* | head -1)

# Step time comparison (last 3 steps)
echo "=== GLOO ===" && grep "Step Time" $GLOO_LOG/run_embodiment.log | tail -3
echo "=== NCCL ===" && grep "Step Time" $NCCL_LOG/run_embodiment.log | tail -3

# Per-component time breakdown (last 2 epochs)
for LOG in "$GLOO_LOG" "$NCCL_LOG"; do
  echo "=== $(basename $LOG) ==="
  grep -oP '(actor/run_training|rollout/generate_one_epoch|env/env_interact_step|step)=[0-9.]+' \
    $LOG/run_embodiment.log | tail -8
done
```

### Weight sync duration from nsys profiles

nsys-rep files are under each node's Ray session dir. Run inside container:

```bash
# Find Actor's profile (on head node)
ACTOR_REP=$(find /tmp/ray/session_latest/logs/nsight/ -name "ActorGroup*.nsys-rep" | head -1)
echo "Actor profile: $ACTOR_REP"

# Extract NVTX summary — look for sync_model_to_rollout duration
nsys stats --report nvtxsum "$ACTOR_REP" 2>/dev/null | grep -i "sync_model"
```

Or open `.nsys-rep` in Nsight Systems GUI — look for `actor/sync_model_to_rollout` NVTX range. Check NCCL row: active kernels = NCCL/IB, empty + DtoH copies = GLOO fallback.

For detailed profiling instructions, see the [Nsight Profiler Guide](https://github.com/chenchaoxu7575/RLinf/blob/feat/nsight-profiler-integration/docs/source-en/rst_source/tutorials/advance/nsight_profiler_guide.rst).

### Expected Results

| Metric | GLOO (Exp A) | NCCL (Exp B) |
|--------|-------------|-------------|
| Weight sync (8.5GB state_dict) | ~17.5s | ~242ms |
| NCCL timeline activity | None (empty) | Active (GDRDMA kernels) |
| DtoH copies during sync | Many (GLOO D2H fallback) | Minimal |
| Step time | ~37s | ~26s |

---

## Adapting to New Machines

### Checklist

1. **Network interfaces**: `ip addr show` — find ethernet interface with routable IPv4
2. **GPU-NIC topology**: `nvidia-smi topo -m` — check GPU-NIC affinity (PIX > NODE > SYS)
3. **IB HCA**: `ibstat` — identify HCA names (e.g. `mlx5_0`)
4. **GPU identification**: `nvidia-smi -L` on both nodes — check if GPU names match

### YAML env_vars to update

| Variable | Purpose | How to determine |
|----------|---------|-----------------|
| `GLOO_SOCKET_IFNAME` | Ethernet for GLOO bootstrap | `ip addr show` — routable IPv4 interface |
| `NCCL_SOCKET_IFNAME` | Ethernet for NCCL bootstrap | Same as above |
| `NCCL_IB_HCA` | IB HCA for data transfer | `ibstat` — pick HCA closest to GPU0 per `nvidia-smi topo -m` |
| `NCCL_NET_GDR_LEVEL` | GPU Direct RDMA level | Set `SYS` if GPU not correctly recognized |
| `RLINF_FORCE_ACCEL_CCL` | Force NCCL for weight sync | Set `1` if GPU model names differ across nodes |

**Important**: The `node1_cpu` group (Env worker) also needs `GLOO_SOCKET_IFNAME`.

### Troubleshooting

**FSDP or channel communication hangs**:
- Cause: NCCL/GLOO bootstrap selects IB link-local address (`fe80::...`)
- Check: `ip addr show` — if IB interfaces start with `ib*`, NCCL will prioritize them
- Fix: Set `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` in ALL node groups (including CPU-only)

**Weight sync extremely slow (seconds instead of ms)**:
- Cause: GPU model name mismatch → RLinf falls back to GLOO/TCP
- Check: `nvidia-smi -L` on both nodes, or look for `hetero_models=True` in logs
- Fix: Set `RLINF_FORCE_ACCEL_CCL: "1"` in GPU node groups

**NCCL GDR disabled**:
- Cause: Unrecognized GPU → NCCL reports wrong PCIe distance
- Check: `NCCL_DEBUG=INFO` shows `GPU Direct RDMA Disabled (distance X > Y)`
- Fix: Set `NCCL_NET_GDR_LEVEL: "SYS"`

**Ray `Too many open files`**:
- Fix: `ulimit -n 65536` before starting Ray
