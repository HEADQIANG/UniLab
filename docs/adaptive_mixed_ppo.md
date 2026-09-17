# 同任务自适应多后端 PPO

本功能继续使用一个 PPO learner：每次 rollout 同时从多个进程中的仿真器收集
on-policy 样本，完整 PPO 更新结束后才改变采样比例。每轮总环境数和 rollout
长度保持不变。算法属于仓库内 `vendor/unilab-rl` 独立包，通过 editable 安装使用。

`--mix-config` 新增 `adaptive` 和 `held_out_backends`。示例：

```yaml
held_out_backends: [mujoco]
adaptive:
  metric: mean_step_reward
  strategy: learning_progress
  interval: 50
  warmup: 100
  ema_alpha: 0.2
  temperature: 1.0
  min_ratio: 0.05
  max_ratio: 0.5
  smoothing: 0.3
  freeze_after: 2000
```

运行前先安装各仿真器的官方依赖和外部 worker。必须从远程仓库运行，且确保
`/home/hpf/.local/bin` 在 PATH 中；不要使用自动同步命令删除其他已安装后端：

```bash
cd /home/hpf/yuelk/UniLab
bash scripts/diagnostics/setup_drake_mixed.sh --run \
uv run --no-sync python scripts/tools/run_with_existing_isaac.py \
  --isaacsim-python /home/hpf/miniconda3/envs/atec/bin/python \
  --isaacgym-python /home/hpf/miniconda3/envs/homie/bin/python3.8 \
  --isaacgym-source /home/hpf/atec/MAPush/isaacgym -- \
uv run --no-sync train --algo ppo --task g1_walk_flat \
  --sim-mix motrix=1,drake=1,mjwarp=1,isaacgym=1,isaacsim=1,genesis=1,newton=1 \
  --mix-config docs/examples/adaptive_g1.yaml \
  algo.num_envs=1024 algo.max_iterations=3000 training.device=cuda:0
```

MuJoCo 被声明为 held-out 后，任何把它加入训练样本的配置都报错。其他后端
必须全部可用，不能把缺失的后端当作运行成功。单 RTX 4090 可能无法同时
容纳多个 GPU 仿真器，具体容量由实际环境数和物理求解器决定。

`mean_step_reward` 按样本数聚合，`mean_episode_reward` 和
`mean_episode_length` 按完整 episode 数聚合，避免不同环境数量引起总 reward
偏差。`learning_progress` 使用每个后端近期归一化 reward 变化幅度；
随机初始超时偏移造成的第一个不完整 episode 不纳入完整 episode 统计，
但其中每个真实 transition 仍然用于 PPO。
`low_reward` 给共同 reward 定义下较弱的后端更多样本。
这些信号是调度启发式，不能保证全局最佳比例，也不使用 MuJoCo 反馈。

warmup 后每 interval 轮更新一次，EMA 和 smoothing 减少抖动。
min_ratio / max_ratio 限制占比，每个后端至少一个环境；实际整数环境数采用
最大余数分配，舍入误差至多一个环境。初始比例也必须满足边界。
完成 freeze_after 轮时冻结最后的比例，之后继续训练同一个策略。
选择较长更新间隔，避免重建成本和 episode 重置影响训练。

手动控制优先并暂停自动调度；清除后恢复自动比例。自动调度不能与 stages
同时使用。缺失、不完整或非有限指标会保留原比例并记录原因。
checkpoint 包含指标窗口、EMA、比例及冻结状态，恢复时校验配置与轮次，
使用原始命令增加 `algo.resume_path=/absolute/model_N.pt` 和新的
`training.log_dir` 即可。具体命令兼容性见 [mixed_ppo.md](mixed_ppo.md)。

`mix_events.jsonl` 记录窗口指标、调度得分、请求比例及实际环境数；
`adaptive_mix.json` 和 `run_summary.json` 保存最终状态。模型最后使用的比例
见 allocation/iteration 日志，最后一次建议比例将在下一轮开始使用。
固定比例、单后端、自适应比例的公平对照与独立 MuJoCo 评估见
[mix_benchmark.md](mix_benchmark.md)。

开发验证：

```bash
cd /home/hpf/yuelk/UniLab
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest \
  tests/training/test_mixed_config.py tests/config/test_mixed_profiles.py \
  vendor/unilab-rl/tests/algos/test_adaptive_mix.py \
  vendor/unilab-rl/tests/algos/test_mixed_ppo.py \
  vendor/unilab-rl/tests/ipc/test_mix_schedule.py -q
```
