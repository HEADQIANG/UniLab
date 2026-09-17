# G1 多仿真器比例实验

此工作流验证一个可证伪的假设：相同 G1WalkFlat 任务、PPO 参数和采样预算下，
多后端训练能否在独立 MuJoCo sim2sim 测试上胜过每一个单后端训练基线。
MuJoCo 不参与训练或自适应控制器的反馈。MuJoCo-Warp 是单独的训练后端，
但它与 MuJoCo 有共同技术来源，因此本实验不代表对完全独立物理实现的盲测。

关联设计：[mixed PPO ADR](ADR-mixed-ppo.md)。先安装
[mixed PPO 运行依赖](mixed_ppo.md)。所有命令从 `/home/hpf/yuelk/UniLab` 执行。
若使用本机独立安装的 Drake runtime，须先完成
[Drake 混合环境安装](drake_mixed_setup.md)，并在下列 `uv run` 命令前加上
`bash scripts/diagnostics/setup_drake_mixed.sh --run`。
这个前缀只设置已安装 runtime 的共享库路径，子进程会继承；不加前缀可能在
构建训练命令时就因 `libfmt.so.9` 等共享库缺失而停止。
复用已有 IsaacGym/IsaacSim 环境时，还需要
[Isaac 运行器](isaac_mixed_probe.md#use-the-existing-runtimes-during-training)；
将整个 benchmark 命令放在该运行器的 `--` 后，并在外层使用 Drake 前缀。
整个实验运行期间必须保留该运行器进程，才能让后续训练和动态重分配找到外部环境。

## 创建、检查和运行

```bash
cd /home/hpf/yuelk/UniLab
uv run --no-sync python -m unilab.training.mix_benchmark plan \
  --study experiments/g1_all_backends \
  --seeds 11,22,33,44,55 \
  --confirmation-seeds 101,102,103,104,105,106,107,108,109,110 \
  --validation-seeds 1001,1002,1003 \
  --test-seeds 2001,2002,2003 \
  --num-envs 2048 --rollout-steps 24 --iterations 2200 \
  --episodes 5 --device cuda:0

uv run --no-sync python -m unilab.training.mix_benchmark preflight \
  --study experiments/g1_all_backends

uv run --no-sync python -m unilab.training.mix_benchmark run \
  --study experiments/g1_all_backends

uv run --no-sync python -m unilab.training.mix_benchmark report \
  --study experiments/g1_all_backends
```

`plan` 只创建 `study.json`，不会开始训练。默认后端是 Motrix、Drake、
MuJoCo-Warp、IsaacGym、IsaacSim、Genesis、Newton，不能把 MuJoCo 加入训练。
默认共有 17 个搜索候选（含自适应探索与冻结比例重训）。上例先进行 85 次
搜索/重训，再将选中的固定混合与七个单后端分别在 10 个新训练 seed 上从头
确认，共另加 80 次训练，总计 165 次；每次 108,134,400 transitions，
另有初始化检查与验证评估。CLI 默认搜索训练 seed 为三个、确认 seed 为十个，
因此不设置 `--seeds` 时默认是 51 + 80 = 131 次训练。完整实验开销较大，
依赖与资源应先通过 `preflight`。预检查在同一批次实际启动全部训练后端，
训练一个 PPO iteration，再把生成的策略加载进 MuJoCo 评估。
缺失依赖、不能满足同任务能力或显存不足会停止执行，绝不静默删除后端。
预检查通过只证明短程运行可用，不证明收敛或多仿真器优势。

可使用 `plan --mix-config <path.yaml>` 指定每个训练后端的 `backends` 设备、
超时和运行选项，以及 `adaptive` 参数；格式见 [mixed PPO](mixed_ppo.md)。
本实验文件只接受 `backends`、`adaptive` 和 `held_out_backends: [mujoco]`；
不接受手动阶段计划。所有生成的训练配置都显式保留 MuJoCo。
默认自适应指标为 `mean_step_reward`，策略为 `learning_progress`，
每 50 iterations 更新，warmup 为 100，最低比例 0.05，最高比例 0.5，
最后约 10% 的训练冻结比例。双后端调试时默认最高比例为 0.8，避免两个 0.5
上限让比例无法变化。很短的 smoke 实验需显式缩短 warmup/interval；计划要求
冻结前至少有两个更新窗口，并且冻结后还剩训练时间，否则会拒绝创建。

使用 `--override 'algo.policy.actor_hidden_dims=[256,128]'` 等参数可给所有
实验统一设置网络、PPO、环境和 reward；预算、随机种子、resume 与路由
字段不能被覆盖。需要调整 mini batches 时，仍须整除每个 rollout 的总样本。
使用 `--fixed 'motrix=0.2,drake=0.1,mjwarp=0.1,isaacgym=0.1,isaacsim=0.1,genesis=0.2,newton=0.2'`
增加先验固定比例，每个混合候选必须保留所有后端的正比例。

`--backends motrix,drake` 可用于显式缩小的调试实验，但报告会标记
`full_backend_study: false`，不能用它证明“全部仿真器”目标已经完成。

## 比较和选择规则

所有单后端通过 `--sim-mix backend=1` 运行，与混合模型共用任务 profile、
PPO learner 和运行机制；不使用原本可能单独调优的单后端 reward 或环境。
搜索阶段所有候选使用相同环境总数、rollout 长度、iteration 数、网络和搜索
训练 seed 集合。最终确认阶段保留相同配置和每次训练预算，但改用独立的
确认训练 seed 集合；选中的固定混合和每个单后端都从头训练，彼此匹配 seed。
GPU 分配、各后端专有物理实现和吞吐量本身不能假定相同。

候选包含每一个单后端、均匀混合、分别偏向各后端 50% 的固定混合、
用户追加的固定混合和自适应混合。先完成自适应探索，以训练 seed 间最终
实际环境比例的平均值生成 `frozen_candidate.json`，再从头训练该固定比例。
每个自适应探索进程必须实际完成至少两次有效更新并进入冻结状态；仅有足够
训练时长但缺少 reward/episode 指标导致更新跳过时，不会生成“已学得比例”。
这样推荐的常数比例有独立重训证据，不把某次控制器的最后输出当作最优解。
自适应搜索的额外计算量不计入固定候选的相同训练预算，并作为限制公开。

所有候选先在相同 MuJoCo validation seeds 上评估。以完整 episode return
的训练 seed 均值选出最佳固定混合（包括冻结重训）并保存 `selection.json`。
自适应训练本身是一个动态策略，因此不会直接被当作最佳常数比例。
选择保存后把比例、训练预算、确认训练 seeds 和测试 seeds 写入
`confirmation_protocol.json`。再在新的确认训练 seeds 上，从头训练选中的
固定混合和每个单后端，最后使用独立 test seeds 评估这些新 checkpoint。
这样避免候选比例选择过拟合到搜索阶段的训练 seed 或具体已训练策略。
搜索训练、确认训练、validation 环境、test 环境四组 seed 不能相交。
不能根据 test 分数再次选择比例；
如果进行下一轮搜索，应预先设置新的独立 test seeds 并创建新的 study。

MuJoCo 使用训练快照中的同一任务、reward、观测和网络配置，加载前检查
sim2sim 合同；缺少快照、维度不符或找不到 checkpoint 都会失败，不回退到零动作。
评估采用 CPU 确定性策略推理、固定 episode seeds、原始任务长度与自然终止。
每个 episode 分别重设随机数，避免较早跌倒改变下一个 episode 的初始指令。
记录 episode return、存活时间占比、完整存活率，以及线速度/角速度追踪得分。
追踪得分是原任务 `exp(-squared_error / sigma)`，已除去 reward 权重，越高越好；
它不是 m/s 单位的误差。环境噪声保持同一训练合同，只固定其随机种子。
评估数据仅写在 benchmark 目录，不会传回 PPO 或控制器。

## 结果和失败恢复

`report.json` 给出搜索阶段选定的固定比例、每个单后端的独立确认测试结果、
按匹配确认训练 seed 计算的差值、描述性 bootstrap 区间和精确单侧配对
sign-flip 检验。episode 不会被当成独立训练重复。bootstrap 区间仅描述
已观察到的 seed 间变异，不用于判定优势，尤其不能用三个正差值的窄区间
宣称显著。精确检验在零假设下假定配对差值的正负符号可交换；使用
Holm 校正同时比较所有单后端，只有每个平均差值都大于零、每个校正后
p 值都不超过 0.05 才标记有优势。

少量确认 seeds 可能在数学上不足以达到显著性：三个差值即使全部为正，
未经校正的最小单侧 p 值仍为 0.125。默认十个确认 seeds 允许检验达到
显著性，但不保证结果有优势；最多支持 40 个确认 seeds 的精确计算。
少于三个确认 seeds 不输出描述性 bootstrap 区间。
`mixed_advantage_not_established` 是合法结果，不能为了支持猜想选择性隐藏它。
报告推荐的是有限候选集合中的最佳已观测比例，不保证全局最优。

记录每次训练从启动到进程退出的 wall time 和 transitions/second，包括初始化、
保存 checkpoint 与重分配开销。相同 transitions 的策略质量比较回答样本效率；
吞吐量回答每秒采样成本。它们不直接证明相同 wall time 下策略质量更高，
此类结论需要额外的相同时间预算或学习曲线实验。

搜索目录为 `<候选>/seed_<seed>/`，最终确认目录为
`confirmation/<候选>/seed_<seed>/`。每次训练目录含 `process.json`、
`console.log` 和 `train/`；原始训练工件
包括 `run_summary.json`、`run_config.json`、`mixed_config.json`、checkpoint
及 `mix_events.jsonl`。搜索验证和确认测试分别保存在各自 trial 下的
`validation/` 和 `test/`。benchmark 会审计实际 allocation 记录：固定候选
必须始终使用声明的比例；任何被接受的手动控制请求都会使该 trial 无效，
避免结果仍标记为“固定均匀比例”而实际采样已被修改。
每次记录的整数环境数必须与分配器对声明比例的计算一致、所有后端均为正数，
并与最终训练汇总一致；训练进程还必须成功退出且具有有效的耗时记录。
即使 checkpoint 已保存，尚未退出或失败的进程也不会被计入完成结果。
生成文件使用原子替换，study 带内容指纹，评估绑定 checkpoint 的 SHA-256。
计划记录 UniLab/unilab-rl、unisim 和已安装 drake_uni 的 Python、配置、
包内 C/C++ 文件和原生 `.so` 扩展、版本控制中 XML 的 SHA-256，同时
保存实际模块路径、可用的 Git 提交、Python/依赖版本及 `direct_url.json` 安装来源。
因此同为 `drake-uni==0.1.0` 的不同源码或构建不会被当作同一引擎。
指纹排除动态生成 XML 和缓存，
`run/preflight` 在这些内容改变后拒绝混入原实验。
IsaacGym/IsaacSim 的外部 worker 环境、包外系统共享库和 GPU 驱动尚未被
自动指纹覆盖，报告元数据会明确说明这一限制；实验期间须保持这些环境
不变并保留外部环境的安装/版本记录。改动它们后应重新建 study。
`run` 可以跳过已成功且身份一致的工作，两个执行器不能同时运行同一 study。
缺失、失败、不完整或过期结果使 `report` 返回退出码 2 和 `incomplete`。
`run` 完成全部步骤后同样审计报告，若审计不完整则返回失败并保留报告供排查。

失败的进程目录会保留，不会自动覆盖或重启。先查看其 `console.log` 和
`process.json`，确认对应进程已终止，再把那个失败目录移到新的备份路径后重跑。
若改动实验参数、代码、依赖或物理后端，应新建 study，避免混用结果；
不要修改已有的 `study.json` 或 `selection.json`。

## 开发检查

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest tests/training/test_mix_benchmark.py -q
uv run --no-sync ruff check src/unilab/training/mix_benchmark.py tests/training/test_mix_benchmark.py
uv run --no-sync ruff format --check src/unilab/training/mix_benchmark.py tests/training/test_mix_benchmark.py
uv run --no-sync mypy src/unilab/training/mix_benchmark.py
uv run --no-sync pyright
```

单元测试覆盖预算、后端完整性、held-out 隔离、选择规则、失败恢复和统计报告，
不等于真实物理引擎集成测试。真实集成使用上面的 `preflight` 与完整 `run`。
