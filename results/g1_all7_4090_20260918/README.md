# G1 七后端正式实验：运行协议与前置校准

截至 2026-09-18 01:46:48（Asia/Shanghai），正式训练已完成 **1/131**：
自适应搜索 seed 11 通过退出、采样预算、比例事件和最终 checkpoint 审计。
seed 22 正在运行（PID 2555679），后台启动 PID 1080640 仍存活；SSH 断开
不会结束作业。此时尚无正式 MuJoCo validation/test 评估，没有比例推荐或
多后端优势结论。原始工件在远程 UniLab 的
`logs/mix_studies/g1_all7_4090_20260918/`。
本目录保存不可变 [study.json](study.json)、首个已完成 trial 的紧凑证据
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
还没有经过三个搜索 seed 的汇总、固定比例重训或独立确认，不能当作推荐比例。

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
