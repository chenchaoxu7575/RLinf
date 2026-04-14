# Blackwell (sm_100/sm_120) Docker Image

支持 B200/B100 (sm_100) 和 RTX 5000 Pro/5090/6000 (sm_120) 的 RLinf 训练镜像。

## 构建

```bash
# 在任意 x86_64 机器上 (不需要目标 GPU)
# build context 需要是 RLinf repo (upstream/main)，提供 pyproject.toml 和 requirements/
cd /path/to/RLinf
docker build -f /path/to/Dockerfile.blackwell -t rlinf-blackwell:latest .

# 导出 sqsh (用于 enroot)
enroot import -o rlinf-blackwell.sqsh "dockerd://rlinf-blackwell:latest"
```

构建时间约 1-2 小时（主要是 flash-attn 编译）。

## 包含的环境

| venv | 用途 | 关键包 |
|------|------|--------|
| openpi | Pi0.5 DSRL 训练 | torch 2.7.1+cu128, flash-attn 2.7.4.post1, openpi 0.1.0 |
| openvla | SAC CNN 训练 | torch 2.7.1+cu128, openvla |

两个 venv 通过 `source switch_env openpi` / `source switch_env openvla` 切换。

## 与官方 RLinf 镜像的区别

| | 官方镜像 | Blackwell 镜像 |
|--|---------|---------------|
| 基础镜像 | nvidia/cuda:12.4.1 | nvidia/cuda:12.8.1 |
| torch | 2.6.0+cu124 (sm_90 max) | 2.7.1+cu128 (sm_120) |
| NCCL | 2.21.5 | 2.28.9 (修复 Blackwell send/recv bug) |
| flash-attn | 预编译 wheel (sm_90) | 交叉编译 (sm_90+sm_100+sm_120) |
| nsight-systems | 无 | 2025.3.1 |
| nccl-rdma-sharp-plugins | 基于 NCCL 2.21 | 基于 NCCL 2.28.9 |

## 已知问题与 Workaround

### 1. UV_TORCH_BACKEND 检测失效
`uv sync` 的 `UV_TORCH_BACKEND=auto` 在 Docker build 环境下无法正确检测 CUDA 版本，会装成 cu124。Dockerfile 中通过精确指定 wheel URL 绕过。

### 2. NCCL 2.26.2 Blackwell bug
torch 2.7.1 自带的 NCCL 2.26.2 在 Blackwell 上 `ncclDevKernel_SendRecv` 越界写入，导致跨节点通信 `illegal memory access`。
- 参考: [PyTorch #170003](https://github.com/pytorch/pytorch/issues/170003), [#152780](https://github.com/pytorch/pytorch/issues/152780)
- 修复: Dockerfile 中升级 `nvidia-nccl-cu12==2.28.9`
- 注意: `torch.cuda.nccl.version()` 仍报 2.26.2（编译时版本），运行时实际用 2.28.9

### 3. NCCL bootstrap 网卡选择
NCCL 2.28.9 bootstrap 默认选 IB link-local 地址 (`fe80::...`)，跨节点不可路由。
- 修复: YAML env_vars 中必须设 `NCCL_SOCKET_IFNAME` 指向以太网网卡

### 4. nccl-rdma-sharp-plugins 版本匹配
升级 NCCL 后 IB plugin 接口从 v10 变为 v11，旧 plugin 不兼容。Dockerfile 中基于 NCCL 2.28.9 重编。

### 5. flash-attn 是 Blackwell 上的硬性依赖
没有 flash-attn 时 transformers 走 SDPA attention 路径，在 Blackwell 上触发 `illegal memory access`。

### 6. openpi 依赖约束
openpi 0.1.0 pin 了 `torch==2.7.1` 和 `numpy<2.0`。torch 升级时 `--no-deps` 防止拉不同版本，之后手动 `pip install numpy==1.26.4`。

### 7. `--force-reinstall` 的陷阱
`pip install torch --force-reinstall` 会卸掉依赖链上的 flash-attn、openpi 等包。Dockerfile 中全部用 `--no-deps`。

## 运行时注意事项

容器启动时需要:
```bash
enroot start --rw \
  --mount /path/to/project:/workspace/rlinf_pub \
  --mount /dev/infiniband:/dev/infiniband \
  <container_name>
```

启动后:
```bash
ulimit -n 65536          # Ray 需要
source switch_env openpi  # 或 openvla
```

宿主机上:
```bash
nvidia-smi -pm 1         # GPU persistence mode, 加速 CUDA 初始化
opensm &                  # 如果 IB 端口不是 Active
```
