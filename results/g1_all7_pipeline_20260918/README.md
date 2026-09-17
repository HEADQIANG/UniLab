# 七后端 G1 PPO 流程验证，2026-09-18

本目录保存一次已完成的机械流程验证：每次训练只有 **448 transitions，
共 25 次训练**，另有一次 112-transition 预检查。它验证七个仿真器共同采样、
自适应比例更新、冻结比例重训、七个单后端对照以及独立 MuJoCo 评估能够连通。
它没有证明学会 G1 行走、多后端训练优势或最佳比例。

[原始报告](report.json) 的状态为 `completed`，结论为
`mixed_advantage_not_established`。确认阶段只有一个训练 seed、一个评估 seed
和一个完整 episode（`n=1`），不能据此作显著性或稳定性能结论。

## 代码与实验协议

- 对应源码提交：`31be361fb0f172626bf5de2027ca7000a15bd809`。
- Study ID：`e64a258b0fe80dab10c81e37a651cc6ed17a402f2fb30c7a331cba4fbd934ced`。
- 任务：`G1WalkFlat`；PPO；actor/critic 隐藏层均为 `[32,32]`。
- 每次训练：28 个环境 × 4 步 rollout × 4 个 PPO iterations = 448 transitions。
- 搜索/比例重训：17 次，训练 seed 11、MuJoCo validation seed 1001。
- 独立确认：选中的固定混合与七个单后端，共 8 次；训练 seed 101、
  MuJoCo test seed 2001。四类 seed 互不相交。
- 训练后端：Motrix、Drake、MuJoCo-Warp、IsaacGym、IsaacSim、Genesis、Newton。
  MuJoCo 不参与训练或自适应指标反馈。

完整参数、依赖版本、实际命令分别保存在 [study.json](study.json)、
每个 trial 的 `process.json` 和 `train/run_config.json`。
源码安装见 [复现指南](../../docs/reproduce_mixed_study.md)，选择与统计协议见
[实验说明](../../docs/mix_benchmark.md)。

远程训练当时仍位于基点 `9e3bb6b814d694d52bfff0f074d0a2b8ede6f032`
之上的未提交工作树，因此原始 JSON 中保留该 Git 提交。归档时按实验同一算法
重新计算源码 SHA-256，确认提交 `31be361f` 的 `unilab` 和 `uni_rl` 源码均与
实验记录相同；见 [归档审计](validation_evidence/archive_audit.json)。
依赖位于该仓库下时，原始 `source_git_commits` 可能返回外围仓库的 HEAD，
不能把它当作依赖上游提交；应结合包版本、安装来源和源码/原生库 hash 阅读。

## 自适应分配和选择

本次缩短实验使用 `mean_step_reward` / `low_reward`，每轮更新，warmup 为 0，
在第 3 个边界冻结。控制器实际完成两次更新，环境总数始终为 28，所有后端
始终获得样本。下表是实际整数环境数，列顺序与原始事件一致：

| 开始使用的 iteration | Motrix | Drake | MJWarp | IsaacGym | IsaacSim | Genesis | Newton |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| 1 | 10 | 10 | 2 | 1 | 1 | 3 | 1 |
| 2，之后冻结 | 8 | 9 | 2 | 4 | 1 | 2 | 2 |

事件原文见 [mix_events.jsonl](adaptive/seed_11/train/mix_events.jsonl)，
冻结状态见 [adaptive_mix.json](adaptive/seed_11/train/adaptive_mix.json)。
`frozen_adaptive` 将末次实际比例 `[8,9,2,4,1,2,2] / 28` 从头重训。
这些整数比例包含小样本舍入，不能把它们理解为调度器连续比例边界改变。

在本次单 seed validation 中，固定混合候选选择了 `frozen_adaptive`
（return 1.085293）；最佳单后端是 IsaacSim（1.169774）。这只是本次机械
验证中的候选选择，不是推荐用于长期训练的比例。

## 独立确认的原始结果

每行均为确认训练 seed 101、MuJoCo test seed 2001、一个 episode。
存活比例是 episode 实际步数除以完整任务 horizon。所有策略均未存活到
完整 horizon，只有 5.5%–6.1%；这些策略尚未表现出学会稳定行走的证据。

| 训练候选 | MuJoCo episode return | 存活比例（%） | 训练进程耗时（秒） | transitions/秒 |
| --- | ---: | ---: | ---: | ---: |
| frozen_adaptive | 0.400816 | 5.7 | 17.92 | 25.01 |
| single_motrix | 0.580107 | 5.5 | 4.50 | 99.47 |
| single_drake | 0.386780 | 5.7 | 6.02 | 74.46 |
| single_mjwarp | 0.435894 | 5.5 | 6.48 | 69.14 |
| single_isaacgym | -0.245123 | 6.1 | 8.55 | 52.39 |
| single_isaacsim | 0.574259 | 5.5 | 15.14 | 29.60 |
| single_genesis | 0.411771 | 5.5 | 11.97 | 37.42 |
| single_newton | 0.197322 | 5.9 | 8.69 | 51.58 |

所有比较的 Holm 校正 p 值均为 1.0；`n=1` 不输出 bootstrap 区间。
混合策略也没有胜过所有单后端。时间包含初始化、保存和退出等开销，极短
训练时这些成本占比很大；不能据此推断长期吞吐或相同时间预算下的策略优势。
完整未舍入数值、追踪指标和 checkpoint hash 见各 trial 的 `metrics.json`
及 [report.json](report.json)。

## 工件和验证证据

归档保留 231 个原始 study JSON/JSONL，原目录结构和 JSON 内容未改写，包括
生成配置、所有训练/评估进程记录、评估指标、训练配置/摘要、比例事件和
自适应状态。远程与本地的文件名及内容汇总 SHA-256 一致。

另保留 [validation_evidence](validation_evidence) 下的测试日志/XML 和
独立七后端正常退出验证的原始摘要、配置、事件及控制台日志：

| 检查 | 保存的结果 |
| --- | --- |
| UniLab 非 slow 测试 | 1750 passed，4 skipped，897 deselected，1 xfailed |
| unilab-rl 非 slow 测试 | 497 passed，8 skipped，3 deselected |
| UniLab `make check` | 格式、lint、Mypy、Pyright 和测试静态检查通过 |
| 独立 all7 cleanup smoke | 448 transitions、两次比例更新、冻结、训练摘要 completed |

两个 JUnit 文件均无 failure/error；UniLab XML 将一个 xfail 同样记在 skipped
中。cleanup 控制台没有 `resource_tracker` 泄漏提示，原进程 PID 596264 在
归档复查时已不存在。监督任务还观察到退出码 0 和本次新增共享内存无残留；
本归档未保存共享内存的完整前后清单。完整 slow/hardware 测试并不包含在
上述非 slow 测试计数中。

## 复现和边界

按复现指南安装源码及外部模拟器后，使用同一套 Drake/Isaac 启动包装器，
将长程示例的计划命令替换为：

```bash
uv run --no-sync python -m unilab.training.mix_benchmark plan \
  --study experiments/g1_all7_pipeline_reproduction \
  --mix-config docs/examples/adaptive_g1_smoke.yaml \
  --seeds 11 --confirmation-seeds 101 \
  --validation-seeds 1001 --test-seeds 2001 \
  --num-envs 28 --rollout-steps 4 --iterations 4 --episodes 1 --device cuda:0 \
  --override 'algo.policy.actor_hidden_dims=[32,32]' \
  --override 'algo.policy.critic_hidden_dims=[32,32]'
```

再针对这个新 study 执行指南中的 `preflight`、`run`、`report`。
这是重跑流程，不能保证不同机器的 GPU 轨迹和数字逐位一致。

本目录不包含 checkpoint、TensorBoard、训练控制台大日志、Git diff 日志、
机器人网格、安装环境和下载的模拟器运行库。因此它是可检查的紧凑结果归档，
不能直接当作可恢复训练目录，也不能直接对这里运行原 CLI 的 `report` 来
重新验证 checkpoint 内容。原始 checkpoint hash、路径和命令完整保留。
Isaac 外部环境、包外系统库和 GPU 驱动未被自动完整指纹化，相关限制保留
在原始 provenance 中。公开长期性能结论仍需要足够训练预算和多个独立确认
seed 的完整实验。
