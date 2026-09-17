# 七后端 G1 自适应 PPO 验证，2026-09-18

七个训练仿真器共用同一个 PPO learner 的完整小规模流程已经跑通：自适应
更新比例、冻结比例重训、各单后端训练、独立 MuJoCo validation/test 和最终
审计报告均完成。此次每次训练只有 448 transitions、四次 PPO 更新，共 25
个训练 trial；它是机械集成验证，尚未建立多后端训练优势或最佳比例结论。

源码提交为 `31be361fb0f172626bf5de2027ca7000a15bd809`。
完整紧凑工件、参数、结果表和复现命令见
[结果归档](../results/g1_all7_pipeline_20260918/README.md)，原始机器报告见
[report.json](../results/g1_all7_pipeline_20260918/report.json)。

## 实际覆盖范围

Motrix、Drake、MuJoCo-Warp、IsaacGym、IsaacSim、Genesis、Newton 均实际参与
混合训练；MuJoCo 始终只用于 sim2sim 评估。各单后端与混合候选使用同一
G1 task、reward、观测/动作合同、PPO、28 个环境、每轮 4 步和 `[32,32]`
actor/critic 网络。搜索训练 seed 11 与确认训练 seed 101 分离；MuJoCo 的
validation seed 1001 与 test seed 2001 分离。

归档审计确认了 25 次等样本预算训练、17 次 validation 评估和 8 次独立确认
test 评估；加上预检查，总计 52 个进程记录均为退出码 0。每个混合 trial
都保留七个后端的正环境数，所有配置均将 MuJoCo 声明为 held out。
自适应探索实际完成两次比例更新并冻结；总环境数始终为 28。

## 结果边界

单 seed validation 在固定混合候选中选择了 `frozen_adaptive`，实际比例为
Motrix/Drake/MJWarp/IsaacGym/IsaacSim/Genesis/Newton =
`[8,9,2,4,1,2,2] / 28`。这个比例被独立从头重训，不是拿探索 checkpoint
直接作为确认模型。

在确认测试中，混合模型 episode return 为 0.400816，Motrix 单后端为
0.580107，IsaacSim 单后端为 0.574259。混合模型没有胜过全部单后端；
原始报告为 `mixed_advantage_not_established`。确认训练 seed 数仅为 1，
所有 Holm 校正 p 值均为 1.0，不输出 bootstrap 区间。所有模型在 MuJoCo
只存活完整 horizon 的 5.5%–6.1%，尚无学会稳定行走的证据。

这些结果验证了流程能够输出失败于原猜想的诚实结论。长期策略质量、
有说服力的多后端优势及可推荐比例仍需更长训练和多个独立确认 seed；
详见 [完整实验协议](mix_benchmark.md)。

## 代码与资源验证

本次归档的 UniLab 非 slow 回归结果为 1750 passed、4 skipped、897
deselected、1 xfailed；unilab-rl 为 497 passed、8 skipped、3 deselected。
`make check` 的格式、lint、Mypy、Pyright 和测试静态检查通过。原始日志及
JUnit XML 位于 [validation_evidence/tests](../results/g1_all7_pipeline_20260918/validation_evidence/tests)。
这些计数不包括完整 slow/hardware 测试矩阵。

另一次七后端 cleanup smoke 完成四次 PPO 更新、两次比例更新并冻结，
摘要为 completed；保存的控制台没有 resource-tracker 泄漏提示，原进程在
归档复查时已不存在。其 [原始证据](../results/g1_all7_pipeline_20260918/validation_evidence/all7_cleanup)
和监督进程观察的范围见 [归档审计](../results/g1_all7_pipeline_20260918/validation_evidence/archive_audit.json)。

远程实验运行在旧基点的未提交工作树；源码内容 hash 与上述发布代码提交
相匹配，原始 Git 元数据未被改写。归档保留原始 JSON 和 checkpoint hash，
没有包含 checkpoint、大型训练日志或外部安装环境，不能直接用于恢复训练。
2026-09-14 的 [历史验证记录](mixed_ppo_validation.md) 保留当时的事实，
本页补充后续七后端实际运行证据。
