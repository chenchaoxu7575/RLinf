# H20-L20 通信配置说明

这份只说明两件事: 异构 GPU 怎么走 NCCL，以及 GLOO 怎么指定高速网卡。当前 H20/L20 机器上的 ResNet 1-rank dummy realworld 实验已经验证通过。

## 1. 异构 NCCL 通信

目标: H20 Actor 和 L20 Rollout 之间同步 GPU 权重时走 NCCL，不因为 GPU 型号不同回退到 GLOO。

### Config 怎么配

H20/L20 的 GPU node group 都要配这些变量:

```yaml
- RLINF_FORCE_ACCEL_CCL: "1"
- NCCL_CROSS_NIC: "1"
- GLOO_SOCKET_IFNAME: "<高速网卡名>"
- NCCL_SOCKET_IFNAME: "<高速网卡名>"
- NCCL_IB_HCA: "<GPU 亲和的 HCA>"
```

当前这两台机器的 1-rank/GPU0 配置:

```yaml
cluster:
  num_nodes: 2
  distributed_channel: true
  component_placement:
    actor:
      node_group: h20
      placement: 0-0
    rollout:
      node_group: l20
      placement: 0-0
    env:
      node_group: l20_node
      placement: 0

  node_groups:
    - label: h20
      node_ranks: 0-0
      env_configs:
        - node_ranks: 0-0
          python_interpreter_path: /opt/venv/openvla/bin/python
          env_vars:
            - RLINF_FORCE_ACCEL_CCL: "1"
            - NCCL_CROSS_NIC: "1"
            - GLOO_SOCKET_IFNAME: "enp26s0f0np0"
            - NCCL_SOCKET_IFNAME: "enp26s0f0np0"
            - NCCL_IB_HCA: "mlx5_0"

    - label: l20
      node_ranks: 1-1
      env_configs:
        - node_ranks: 1-1
          python_interpreter_path: /opt/venv/openvla/bin/python
          env_vars:
            - RLINF_FORCE_ACCEL_CCL: "1"
            - NCCL_CROSS_NIC: "1"
            - GLOO_SOCKET_IFNAME: "ens1006f0np0"
            - NCCL_SOCKET_IFNAME: "ens1006f0np0"
            - NCCL_IB_HCA: "mlx5_0"
```

CPU/env worker 不要配 NCCL，只配 GLOO:

```yaml
- label: l20_node
  node_ranks: 1-1
  ignore_hardware: true
  env_configs:
    - node_ranks: 1-1
      python_interpreter_path: /opt/venv/openvla/bin/python
      env_vars:
        - GLOO_SOCKET_IFNAME: "ens1006f0np0"
```

当前完整配置文件:

```bash
examples/embodiment/config/realworld_dummy_turtle2_sac_cnn_2node1rank_h20_l20_nccl_async.yaml
```

多卡时只需要按拓扑改 `placement` 和 `NCCL_IB_HCA`。例如 H20 4 卡可能是:

```yaml
placement: 0-3
NCCL_IB_HCA: "mlx5_0,mlx5_1,mlx5_2,mlx5_5"
```

L20 4 卡按已有记录通常是:

```yaml
placement: 0-3
NCCL_IB_HCA: "mlx5_0,mlx5_1"
```

### 代码怎么改

改 `rlinf/scheduler/collective/multi_channel_pg.py`。

原逻辑: 只要同一个 process group 里 GPU model 不同，比如 H20 + L20，就禁用 accelerator CCL。

新增开关:

```python
force_accel_ccl = os.getenv("RLINF_FORCE_ACCEL_CCL", "0") == "1"
```

force 只允许在这些条件都满足时生效:

- 所有 worker 都是真实 GPU worker，不是 CPU/no-accelerator。
- worker 的 accelerator type 相同，比如都是 NVIDIA GPU。
- backend 在 RLinf 支持的 CCL 列表里。
- worker name 不是 `Channel` / `Metric`。

核心判断:

```python
hetero_models = any(w.accelerator_model != accel_model for w in group_info.workers)
same_accel_type = all(w.accelerator_type == accel_type for w in group_info.workers)
all_have_accel = all(
    w.accelerator_type != AcceleratorType.NO_ACCEL
    and w.accelerator_model != ""
    and "Channel" not in w.address.get_name()
    and "Metric" not in w.address.get_name()
    for w in group_info.workers
)
supported_accel_ccl = accel_type in AcceleratorUtil.CCL_SUPPORT_LIST
force_allowed = (
    force_accel_ccl and all_have_accel and same_accel_type and supported_accel_ccl
)

self._no_accel_ccl = (
    (hetero_models and not force_allowed)
    or accel_type == AcceleratorType.NO_ACCEL
    or not same_accel_type
    or not supported_accel_ccl
)
```

可选开关:

```bash
RLINF_NCCL_EAGER_INIT=1
```

只在排查 NCCL lazy init hang 时打开，默认不建议开，因为会增加显存占用。

## 2. 高速网卡 GLOO

目标: Env、Channel 这类 CPU/小消息通信走 GLOO，但不要走管理网，要走 H20/L20 的高速 NIC。

### Config 怎么配

所有会参与 GLOO 的 node group 都显式设置:

```yaml
GLOO_SOCKET_IFNAME: "<高速网卡名>"
```

这包括:

- H20 GPU node group
- L20 GPU node group
- L20 CPU/env node group

当前两台机器:

```yaml
H20: GLOO_SOCKET_IFNAME: "enp26s0f0np0"
L20: GLOO_SOCKET_IFNAME: "ens1006f0np0"
```

同时打开:

```yaml
cluster:
  distributed_channel: true
```

这样每个节点都有本地 ChannelWorker，小消息不会全部绕到 head 节点。

### 代码怎么改

改 runner 创建 channel 的地方，把 YAML 的 `distributed_channel` 传进去。

`rlinf/runners/embodied_runner.py`:

```python
distributed_channel = self.cfg.cluster.get("distributed_channel", False)

self.env_channel = Channel.create("Env", distributed=distributed_channel)
self.rollout_channel = Channel.create("Rollout", distributed=distributed_channel)
self.actor_channel = Channel.create("Actor", distributed=distributed_channel)
```

### GLOO 样例

只关心 GLOO 走高速网卡时，可以参考同目录:

```bash
codex_notes/hetero_nccl/gloo_fast_nic_example.yaml
```

最小片段:

```yaml
cluster:
  num_nodes: 2
  distributed_channel: true
  node_groups:
    - label: h20
      node_ranks: 0-0
      env_configs:
        - node_ranks: 0-0
          env_vars:
            - GLOO_SOCKET_IFNAME: "enp26s0f0np0"
    - label: l20
      node_ranks: 1-1
      env_configs:
        - node_ranks: 1-1
          env_vars:
            - GLOO_SOCKET_IFNAME: "ens1006f0np0"
    - label: l20_node
      node_ranks: 1-1
      ignore_hardware: true
      env_configs:
        - node_ranks: 1-1
          env_vars:
            - GLOO_SOCKET_IFNAME: "ens1006f0np0"
```

不要把 `RLINF_FORCE_ACCEL_CCL=1` 用在纯 GLOO baseline 里；这个开关是给异构 NCCL 权重同步用的。

