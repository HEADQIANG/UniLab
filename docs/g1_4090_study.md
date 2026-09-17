# G1 七后端单卡实验协议

这份有固定预算的实验沿用 [完整公平比较协议](mix_benchmark.md)，在一张
RTX 4090 上依次训练候选。七个训练后端同时向同一个 PPO learner 采样；
MuJoCo 只用于选择和独立评估。安装及外部 Isaac 路径设置见
[复现指南](reproduce_mixed_study.md)。

每次训练保持 224 个环境、24 步 rollout、1200 次 PPO 更新，共
6,451,200 transitions，actor/critic 都使用任务原有的 `[512,256,128]`
网络和同一组 PPO 参数。搜索训练 seeds 为 11、22、33；固定比例选定后，
使用 101–110 十个新训练 seeds 从头确认该混合与七个单后端。
总计 51 次搜索/冻结重训及 80 次独立确认，即 845,107,200 个训练样本，
另加预检查及评估。每个 checkpoint 在相应 split 的三个 MuJoCo seeds
上各评估五个完整 episode。测试 seeds 与先前机械流程验证不同。

这是低于任务默认大规模 preset 的预算，不预先保证训练收敛或混合优势。
最终必须同时报告学习程度、所有单后端比较、配对统计及时间成本。
不能从 448 样本的集成测试中直接推荐正式训练比例。

作为预算参照，项目 [G1 MuJoCo owner](../src/unilab/conf/ppo/task/g1_walk_flat/mujoco.yaml)
默认使用 2048 个环境、2200 次更新，配合 [PPO 默认配置](../src/unilab/conf/ppo/config.yaml)
的 24 步 rollout，共 108,134,400 transitions。本研究单次训练约为该预算的
5.97%；默认配置本身也不是已收敛的证据。

报告的 `mixed_advantage_supported_at_equal_samples` 检验的是 episode return
的相对优势，没有内置绝对行走质量门槛。必须同时检查 `survived_horizon`
和 `survival_fraction`；如果所有方案都在短时间内终止，即使回报差异显著，
也只能报告该预算下的相对改进，不能据此宣称可靠行走。带探索噪声的训练
回合统计与确定性策略评估应分开解释。

训练日志的 `Metrics/twist/error_vel_xy`、`error_vel_yaw` 将累计误差除以
固定命令周期的步数，短回合中的值不能直接解释为实际每步平均误差。
正式评估中的 `tracking_lin_vel`、`tracking_ang_vel` 则是去除权重后的
追踪奖励得分，也不是 m/s 或 rad/s 单位的误差，见 [指标说明](mix_benchmark.md)。

自适应配置见 [adaptive_g1_4090.yaml](examples/adaptive_g1_4090.yaml)：
以训练 reward 的学习进展调整比例，每 100 次 PPO 更新观察一次，100 次
warmup，第 1100 次后冻结；每个后端的请求比例限制在 5%–50%，实际整数
环境数可能产生舍入差异。每个观察
窗口含 2400 个环境时间步，超过任务的 1000 步 episode 时限。
整数环境分配及物理状态重建仍按 mixed PPO 的公开规则处理并记录。

在仓库根目录运行（先按复现指南设置三个 `STUDY_ISAAC*` 路径变量）：

```bash
uv run --no-sync python -m unilab.training.mix_benchmark plan \
  --study logs/mix_studies/g1_all7_4090_20260918 \
  --mix-config docs/examples/adaptive_g1_4090.yaml \
  --seeds 11,22,33 \
  --confirmation-seeds 101,102,103,104,105,106,107,108,109,110 \
  --validation-seeds 3001,3002,3003 --test-seeds 4001,4002,4003 \
  --num-envs 224 --rollout-steps 24 --iterations 1200 \
  --episodes 5 --device cuda:0

bash scripts/diagnostics/setup_drake_mixed.sh --run \
  uv run --no-sync python scripts/tools/run_with_existing_isaac.py \
  --isaacsim-python "$STUDY_ISAACSIM_PYTHON" \
  --isaacgym-python "$STUDY_ISAACGYM_PYTHON" \
  --isaacgym-source "$STUDY_ISAACGYM_SOURCE" -- \
  uv run --no-sync python -m unilab.training.mix_benchmark run \
  --study logs/mix_studies/g1_all7_4090_20260918

uv run --no-sync python -m unilab.training.mix_benchmark report \
  --study logs/mix_studies/g1_all7_4090_20260918
```

`run` 自动先执行全部后端的真实预检查。正式训练期间保持代码、依赖、
外部 Isaac 环境和驱动不变。完整工件留在远程 study 目录；失败目录保留
排查，不自动覆盖。进程结束后审计通过才计为完成，正在运行时报告
`incomplete` 是预期行为。不得依据 test 结果重新选择比例。
