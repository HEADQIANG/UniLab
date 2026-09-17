# G1 七后端正式实验：运行协议与阶段性结果

截至 2026-09-18 05:14:01（Asia/Shanghai），正式训练已完成 **15/131**：
自适应搜索、Motrix、Drake、MJWarp 和 IsaacGym 单后端的 seed 11、22、33
均通过训练审计。
三种子汇总的 `frozen_adaptive` 仍是待重训、待验证候选，不是推荐或最佳比例。
Motrix、Drake、MJWarp 和 IsaacGym 各三个训练种子分别完成 15 个 MuJoCo
validation 回合，共 12 个策略、180 个回合；正式 test 尚未开始。
IsaacSim seed 11 正在运行（PID 4122519，已观测到第 129/1200 次更新）；
后台启动 PID 1080640 仍存活；
SSH 断开不会结束作业。目前没有多后端优势结论。原始工件在远程 UniLab 的
`logs/mix_studies/g1_all7_4090_20260918/`。
本目录保存不可变 [study.json](study.json)、十五个已完成 trial 的紧凑证据
及前置校准；不能把这个带观测时间的状态当作全部实验已完成的结果。

[执行步骤和完整预算](../../docs/g1_4090_study.md)：51 次搜索/冻结重训加
80 次独立确认，每次 6,451,200 transitions，总训练预算 845,107,200。
所有混合方案使用全部七个后端；七个单后端分别对照；MuJoCo 保留评估。
代码实现提交为 `31be361fb0f172626bf5de2027ca7000a15bd809`，本次提交另加
固定实验配置和记录。远程仍保留原基线上的工作树，实际实现由协议中的
源码 SHA-256 绑定。没有把失败或未完成训练计为结果。

## 首个完成的正式 trial：adaptive / seed 11

[进程回执](adaptive/seed_11/process.json) 记录退出码 0，完整进程耗时
2651.506030005001 秒；[训练摘要](adaptive/seed_11/train/run_summary.json)
记录 1200 次 PPO 更新和 6,451,200 个实际 transitions。逐轮事件复算也得到
相同采样总数：每轮 5376 个样本、全部七个后端始终正采样。
控制器产生 10 次更新、9 次实际重配置，最终保持冻结。

末尾实际环境数按 Motrix/Drake/MJWarp/IsaacGym/IsaacSim/Genesis/Newton
顺序为 `28/26/25/27/53/37/28`，总计 224。这是该搜索轨迹的最后一次分配，
三种子汇总候选见下节；固定比例重训和独立确认仍待完成，不能当作推荐比例。

[completed_trial_audit.json](adaptive/seed_11/completed_trial_audit.json)
保存观测时间、现有 `training_result` 审计结果、原始文件 SHA-256、事件计数
及最终 checkpoint 的校验值。模型留在远程，未复制进此结果目录。
监督任务另以只读方式验证 checkpoint 可安全加载、浮点参数有限且相对
初期 checkpoint 已变化；这仅是数值/更新检查，不能代替正式 MuJoCo 评估。

比例事件原文采用 [mix_events.jsonl.gz](adaptive/seed_11/train/mix_events.jsonl.gz)
归档，以避免每个 trial 的逐轮 JSONL 膨胀。使用 `gzip -n -9` 压缩；原始
3,056,138 字节变为 451,129 字节，解压后的字节和 SHA-256 与远程原文一致。
从仓库根目录校验或解压（`-k` 保留压缩文件）：

```bash
gzip -dc results/g1_all7_4090_20260918/adaptive/seed_11/train/mix_events.jsonl.gz | sha256sum
# 8f14d32a35003ccc6b355967087f8a1557f502c29f9eb609a59dc193f84790ff
gzip -dk results/g1_all7_4090_20260918/adaptive/seed_11/train/mix_events.jsonl.gz
```

归档还保留原始 `run_config.json`、`mixed_config.json` 和 `adaptive_mix.json`。
缺少模型及完整运行目录时，不能对本地紧凑归档直接重跑完整 checkpoint
审计或恢复训练；上述解压只恢复比例事件文件，不补回省略的工件。

## 第二个完成的正式 trial：adaptive / seed 22

[进程回执](adaptive/seed_22/process.json) 记录退出码 0，完整进程耗时
2583.0349342209993 秒。原始摘要和逐轮事件分别确认 1200 次 PPO 更新、
6,451,200 个实际 transitions；每轮 5376 个样本，七个后端持续正采样。
控制器完成 10 次更新、9 次实际重配置并冻结。

末尾实际环境数按同一后端顺序为 `29/32/26/31/40/18/48`，总计 224。
这仍是单条自适应搜索轨迹的最终分配，不是经独立确认的推荐比例。
seed 22 归档观测时（02:27:12），正式 MuJoCo validation/test 工件均未产生。

[完成审计](adaptive/seed_22/completed_trial_audit.json) 保存带时间的远程
`training_result` 检查、checkpoint hash 和六个原始文件的完整校验值。
未复制模型，也没有根据训练过程数值添加策略质量或多后端优势结论。
原始 JSON 保持不变；[比例事件](adaptive/seed_22/train/mix_events.jsonl.gz)
由 3,064,115 字节无损压缩至 448,746 字节，可按以下命令校验或解压：

```bash
gzip -dc results/g1_all7_4090_20260918/adaptive/seed_22/train/mix_events.jsonl.gz | sha256sum
# f71639ba01fc448b1749b07c2733d949408180b1cd02729f3cf15b958fed65a6
gzip -dk results/g1_all7_4090_20260918/adaptive/seed_22/train/mix_events.jsonl.gz
```

## 第三个完成的正式 trial：adaptive / seed 33

[进程回执](adaptive/seed_33/process.json) 记录退出码 0，完整进程耗时
2631.6879797209986 秒。训练摘要与逐轮事件再次独立确认 1200 次 PPO 更新、
6,451,200 个实际 transitions；每轮 5376 个样本，七个后端始终正采样。
控制器完成 10 次更新、9 次实际重配置并冻结，末尾实际环境数为
`27/20/25/29/60/42/21`，总计 224。

[完成审计](adaptive/seed_33/completed_trial_audit.json) 保存观测时间、现有
`training_result` 审计、最终模型 hash 及原始文件校验值；模型仍留在远程。
[事件原文压缩包](adaptive/seed_33/train/mix_events.jsonl.gz) 保留全部 1230
条记录，3,061,398 字节压缩为 451,227 字节。校验和解压命令如下：

```bash
gzip -dc results/g1_all7_4090_20260918/adaptive/seed_33/train/mix_events.jsonl.gz | sha256sum
# 0c3ebaad95cf17c085dea314e25eccf70f5b2b71b2ab53881d532e7b979c14c1
gzip -dk results/g1_all7_4090_20260918/adaptive/seed_33/train/mix_events.jsonl.gz
```

## 三个搜索 seed 的汇总候选

[frozen_candidate.json](frozen_candidate.json) 是调度器生成文件的原始字节，
没有手工改写。它依次引用 seed 11、22、33 最终模型的 SHA-256，均已在
远程重新核对。候选比例按三个 seed 末尾实际整数环境数除以 224 后取算术
平均，等价于各列环境数之和除以 672；未用 MuJoCo 评估反馈选择这些权重。

| 后端 | seed 11 环境数 | seed 22 环境数 | seed 33 环境数 | 汇总候选比例（%） |
| --- | ---: | ---: | ---: | ---: |
| Motrix | 28 | 29 | 27 | 12.5000 |
| Drake | 26 | 32 | 20 | 11.6071 |
| MuJoCo-Warp | 25 | 26 | 25 | 11.3095 |
| IsaacGym | 27 | 31 | 29 | 12.9464 |
| IsaacSim | 53 | 40 | 60 | 22.7679 |
| Genesis | 37 | 18 | 42 | 14.4345 |
| Newton | 28 | 48 | 21 | 14.4345 |

完整未舍入比例、三模型来源及复算步骤见 [discovery_audit.json](discovery_audit.json)。
三次搜索共采集 19,353,600 transitions；它们只完成了自适应候选生成阶段。
`frozen_adaptive` 还需要从头固定比例重训，并与其他混合候选及七个单后端
完成共同 validation 比较；后续才选择比例并开展独立确认测试。当前候选
不能称为推荐比例、最佳比例或多后端优势证据。

监督任务在 seed 33 最后重配置后、结束前，使用现有 Drake 包装器只读核对
主项目、算法、UniSim 和 Drake 的源码/原生库指纹仍与原协议一致；此项
不覆盖完整的外部 Isaac 环境、系统共享库和 GPU 驱动。原始 study 未修改。

## 首个已完成的单后端验证：Motrix / seed 11

[训练及验证审计](single_motrix/seed_11/completed_trial_audit.json) 对应固定
224 个 Motrix 环境、1200 次更新和 6,451,200 个实际 transitions。
训练进程退出码为 0，完整耗时 307.2626294429974 秒；没有自适应比例调整。
保存的全部 `config.algo` 字段及 sim2sim 合同快照与 `adaptive/seed_11`
完全相等，包括 24 步 rollout 和 `[512,256,128]` actor/critic 网络。

[原始 MuJoCo 验证记录](single_motrix/seed_11/validation/metrics.json) 使用
validation seeds 3001、3002、3003，每个 seed 五个完整 episode；
[评估进程回执](single_motrix/seed_11/validation/process.json) 记录退出码 0。
15 个回合的来源 checkpoint、评估 identity 和逐回合指标均通过复核：

| 指标 | 结果 |
| --- | ---: |
| 平均 episode return | 22.1336813 |
| 平均存活时间占比 | 73.5467% |
| 完整存活 20 秒 | 8/15（53.3333%） |
| 平均线速度追踪得分 | 0.5365561 |
| 平均角速度追踪得分 | 0.5857145 |

追踪得分不是 m/s 或 rad/s 误差，指标口径见
[实验说明](../../docs/g1_4090_study.md)。这是一个训练 seed 的 validation
结果，15 个回合不是 15 次独立训练重复；尚不能据此推断跨 seed 稳定性、
固定比例优劣或多后端优势，也不是独立确认 test 的结果。
完整逐轮训练事件仍以 gzip 原文保存；解压命令为：

```bash
gzip -dk results/g1_all7_4090_20260918/single_motrix/seed_11/train/mix_events.jsonl.gz
```

## Motrix 三个训练 seed 的 validation 结果

新增 [seed 22 审计](single_motrix/seed_22/completed_trial_audit.json) 和
[seed 33 审计](single_motrix/seed_33/completed_trial_audit.json)，保留各自的
原始训练配置、进程回执、逐轮事件以及 15 个 MuJoCo validation 回合。
每次训练均完成 1200 次更新和 6,451,200 transitions；三个种子的完整
`config.algo` 与 sim2sim 合同分别和同种子的自适应训练完全相等。

| 训练 seed | 训练进程耗时（秒） | 平均 episode return | 平均存活时间占比 | 完整存活 20 秒 |
| --- | ---: | ---: | ---: | ---: |
| 11 | 307.2626 | 22.1336813 | 73.5467% | 8/15 |
| 22 | 306.4071 | 21.8703986 | 78.6933% | 7/15 |
| 33 | 304.2063 | 28.6143521 | 98.1200% | 14/15 |

[三种子汇总](single_motrix/discovery_validation_summary.json) 按训练种子
等权计算，平均 return 为 24.2061440，平均存活占比为 83.4533%。
45 个评估回合中 29 个完整存活 20 秒，但这里只包含 **3 个独立训练重复**；
45 回合不能作为 45 个独立训练样本用于统计推断。结果来自 discovery
种子的 validation split，尚无混合候选的共同验证，也未开展新种子的
confirmation/test，不能据此得出比例优劣或多后端优势结论。

从仓库根目录恢复新增种子的逐轮事件（保留压缩原文）：

```bash
for seed in 22 33; do
  gzip -dk "results/g1_all7_4090_20260918/single_motrix/seed_${seed}/train/mix_events.jsonl.gz"
done
```

## Drake 三个训练 seed 的 validation 结果

[seed 11 审计](single_drake/seed_11/completed_trial_audit.json)、
[seed 22 审计](single_drake/seed_22/completed_trial_audit.json) 和
[seed 33 审计](single_drake/seed_33/completed_trial_audit.json) 分别确认固定
224 个 Drake 环境、1200 次连续更新、6,451,200 个实际 transitions。
三个训练进程退出码均为 0。保存的完整 `config.algo` 和 sim2sim 合同
分别与同种子的自适应训练完全相等；
没有比例调整或人工控制事件。

[seed 11 验证记录](single_drake/seed_11/validation/metrics.json)、
[seed 22 验证记录](single_drake/seed_22/validation/metrics.json) 和
[seed 33 验证记录](single_drake/seed_33/validation/metrics.json) 都使用
validation seeds 3001、3002、3003，每个评估 seed 五个 episode。
三个评估进程退出码均为 0；checkpoint 身份和逐回合指标均通过复核。

| 指标 | seed 11 | seed 22 | seed 33 |
| --- | ---: | ---: | ---: |
| 训练进程耗时（秒） | 1299.5465 | 1305.6420 | 1302.2268 |
| 平均 episode return | 18.1274279 | 15.9312872 | 14.5622786 |
| 平均存活时间占比 | 66.0933% | 53.4267% | 48.3467% |
| 完整存活 20 秒 | 9/15（60.0000%） | 3/15（20.0000%） | 1/15（6.6667%） |
| 平均线速度追踪得分 | 0.5719760 | 0.5597717 | 0.5604072 |
| 平均角速度追踪得分 | 0.5700145 | 0.5493818 | 0.5683738 |

[三种子汇总](single_drake/discovery_validation_summary.json) 按训练种子
等权计算，平均 return 为 16.2069979，平均存活占比为 55.9556%，
共 13/45 个回合完整存活 20 秒。这里是 3 个独立训练重复，不能将
45 个回合当作 45 次独立训练；完整存活的回合数存在明显种子间差异。
与其他方案的完整比较及新种子的独立确认仍待完成，当前结果不证明
稳定行走或混合后端优势。追踪指标是奖励得分，
不是物理单位误差。模型权重仍保存在远端，紧凑归档不包含模型。
从仓库根目录解压逐轮事件：

```bash
for seed in 11 22 33; do
  gzip -dk "results/g1_all7_4090_20260918/single_drake/seed_${seed}/train/mix_events.jsonl.gz"
done
```

## MJWarp 三个训练 seed 的 validation 结果

[seed 11 审计](single_mjwarp/seed_11/completed_trial_audit.json)、
[seed 22 审计](single_mjwarp/seed_22/completed_trial_audit.json) 和
[seed 33 审计](single_mjwarp/seed_33/completed_trial_audit.json) 分别确认固定
224 个 MJWarp 环境、1200 次连续更新和 6,451,200 个实际 transitions。
三个训练进程退出码均为 0；完整 `config.algo` 和 sim2sim 合同分别与
同种子的自适应训练相等，样本预算和 PPO 条件一致。

[seed 11 验证记录](single_mjwarp/seed_11/validation/metrics.json)、
[seed 22 验证记录](single_mjwarp/seed_22/validation/metrics.json) 和
[seed 33 验证记录](single_mjwarp/seed_33/validation/metrics.json) 都使用
validation seeds 3001、3002、3003，每个评估 seed 五个 episode。
三个评估进程退出码均为 0；来源 checkpoint、逐回合身份和全部指标均通过复核。

| 指标 | seed 11 | seed 22 | seed 33 |
| --- | ---: | ---: | ---: |
| 训练进程耗时（秒） | 293.2761 | 292.0169 | 293.3067 |
| 平均 episode return | 10.0858040 | 11.6920640 | 1.9562278 |
| 平均存活时间占比 | 32.2133% | 35.5933% | 6.0933% |
| 完整存活 20 秒 | 2/15（13.3333%） | 3/15（20.0000%） | 0/15（0.0000%） |
| 平均线速度追踪得分 | 0.6420331 | 0.6134025 | 0.7241138 |
| 平均角速度追踪得分 | 0.5925886 | 0.5934995 | 0.5349956 |

[三种子汇总](single_mjwarp/discovery_validation_summary.json) 按训练种子
等权计算，平均 return 为 7.9113653（种子间样本标准差 5.2194602），
平均存活占比为 24.6333%（样本标准差 16.1448 个百分点）。
45 个回合中 5 个完整存活 20 秒；这里只包含 **3 个独立训练重复**。
seed 33 的低回报和 0/15 完整存活结果原样保留，没有排除任何训练种子。
这些 discovery validation 结果尚不能证明稳定行走；混合候选的共同验证
和新种子的独立确认仍未完成，没有比例推荐或多后端优势结论。
耗时和质量分别记录，较短训练时间不代表策略质量更好；追踪指标是奖励得分。
模型权重保留在远端，紧凑归档不包含模型。
从仓库根目录解压逐轮事件：

```bash
for seed in 11 22 33; do
  gzip -dk "results/g1_all7_4090_20260918/single_mjwarp/seed_${seed}/train/mix_events.jsonl.gz"
done
```

## IsaacGym 三个训练 seed 的 validation 结果

[seed 11 审计](single_isaacgym/seed_11/completed_trial_audit.json)、
[seed 22 审计](single_isaacgym/seed_22/completed_trial_audit.json) 和
[seed 33 审计](single_isaacgym/seed_33/completed_trial_audit.json) 分别确认
固定 224 个 IsaacGym 环境、1200 次连续更新和 6,451,200 个实际 transitions。
三个训练进程退出码均为 0；完整 `config.algo` 和 sim2sim 合同分别与
同种子的自适应训练相等。

[seed 11 验证记录](single_isaacgym/seed_11/validation/metrics.json)、
[seed 22 验证记录](single_isaacgym/seed_22/validation/metrics.json) 和
[seed 33 验证记录](single_isaacgym/seed_33/validation/metrics.json) 都使用
validation seeds 3001、3002、3003，每个 seed 五个 episode。
三个评估进程退出码均为 0；来源 checkpoint、逐回合身份和指标均通过核验。

| 指标 | seed 11 | seed 22 | seed 33 |
| --- | ---: | ---: | ---: |
| 训练进程耗时（秒） | 472.3726 | 470.1375 | 469.1011 |
| 平均 episode return | 21.2529341 | 21.1222248 | 13.1358678 |
| 平均存活时间占比 | 79.2400% | 68.9267% | 40.1667% |
| 完整存活 20 秒 | 10/15（66.6667%） | 9/15（60.0000%） | 1/15（6.6667%） |
| 平均线速度追踪得分 | 0.5176847 | 0.5641866 | 0.5681762 |
| 平均角速度追踪得分 | 0.5485756 | 0.5740212 | 0.5719129 |

[三种子汇总](single_isaacgym/discovery_validation_summary.json) 按训练种子
等权计算，平均 return 为 18.5036756（种子间样本标准差 4.6491172），
平均存活占比为 62.7778%（样本标准差 20.2494 个百分点）。
45 个回合中 20 个完整存活 20 秒；这里只包含 **3 个独立训练重复**。
三个种子全部纳入统计，seed 33 的较低回报和 1/15 完整存活结果原样保留。
这些是 discovery validation 结果，不能证明跨种子可靠行走。
混合候选验证及新种子的独立确认仍待完成，没有比例推荐或多后端优势结论。
追踪指标是奖励得分，不是物理单位误差。模型权重保留在远端；
从仓库根目录解压逐轮训练事件：

```bash
for seed in 11 22 33; do
  gzip -dk "results/g1_all7_4090_20260918/single_isaacgym/seed_${seed}/train/mix_events.jsonl.gz"
done
```

## 规模校准（独立于正式训练）

均匀七后端、seed 7、224 环境、24 步 rollout、100 次 PPO 更新，使用
任务默认的 `[512,256,128]` actor/critic。共 537,600 transitions，训练
摘要记录 244.2518 秒（含初始化、训练及退出），约 2201 样本/秒。实测
显存约 7.4 GB。这个吞吐不能直接外推到所有单后端或全部自适应阶段。
各后端前十次至末十次的平均训练 step reward 都有所改善，但尚未学会行走。

独立校准 MuJoCo seeds 7001、7002、7003，每个 seed 三个完整 episode，
确定性 CPU 策略；没有使用正式的 validation/test seeds。以下是同一条
训练轨迹的 checkpoint，不能当作五个独立训练重复。原始 episode 记录、
checkpoint SHA-256 见 [校准评估](calibration/calibration_evaluation.json)。

| PPO 更新后的 checkpoint | 平均 episode return | 存活时间占完整时限的比例 |
| --- | ---: | ---: |
| 0 | 0.382450 | 5.6667% |
| 25 | 0.188777 | 1.8889% |
| 50 | 0.401581 | 2.4000% |
| 75 | 0.441404 | 2.4889% |
| 99 | 0.619424 | 2.9556% |

所有校准 episode 均未存活到完整 20 秒时限。末尾 return 提高但存活比例
低于最初 checkpoint，因此不能由 reward 改善宣称策略已稳定或任务已解决。
此校准只确认运行规模可行并记录早期学习情况，不参与正式比例选择。

## 七后端断点续训

从四次更新的自适应机械复验 checkpoint 恢复，再训练两次，最终
`next_iteration=6`，控制器保持冻结；环境数保持 Motrix/Drake/MJWarp/
IsaacGym/IsaacSim/Genesis/Newton = `6/2/1/1/5/7/6`。恢复后的采样事件与
配置保存在 [resume/](resume/)，证据不表示物理轨迹逐位恢复：环境物理
状态会重新初始化，PPO/优化器/控制器状态按 checkpoint 恢复。

本目录不包含模型权重、完整 stdout、TensorBoard 或下载的仿真运行库。
这些仍保存在远程原始目录；重新训练需要按复现指南安装相同依赖和资产。

## 外部 Isaac 运行环境补充记录

正式训练启动后以只读方式记录了外部环境的安装清单：
[IsaacGym](external_runtime/isaacgym_packages.json) 使用 Python 3.8.20、
IsaacGym 1.0rc4、Torch 2.0.1+cu118；
[IsaacSim](external_runtime/isaacsim_packages.json) 使用 Python 3.11.15、
IsaacSim 5.1.0.0、IsaacLab 0.54.3、Torch 2.7.0+cu128。
外部 worker 只负责仿真；共享 PPO learner 使用主项目的 Torch 2.8.0+cu128。

[主机与选定源文件记录](external_runtime/host_and_selected_sources.json)
包含驱动 580.159.03、IsaacLab Git 提交及其 tracked 工作树状态，以及
七个 IsaacGym 源文件/核心原生绑定的 SHA-256。安装包清单通过
`importlib.metadata` 读取，没有导入模拟器、安装软件包或修改环境。
这些是带采集时间的补充证据，不是外部环境/系统库/JIT 缓存/资产的完整
指纹，也不替代或修改原始不可变 study ID。
