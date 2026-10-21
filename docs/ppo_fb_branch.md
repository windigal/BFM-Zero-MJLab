# PPO-FB 分支

更新：2026-10-22。工作总结与训练结果见 [WORK_SUMMARY.md](../WORK_SUMMARY.md)，
后续任务见 [PLAN.md](../PLAN.md)，换电脑恢复见 [ppo_fb_setup.md](ppo_fb_setup.md)。
初始设计见 [分析文档](ppo_fb_controller_analysis.md)。实现继承 rsl-rl 5.2.0，
不修改依赖源码。0804/0810 均可作为起点，默认 0810。数据默认相对路径为
`humanoidverse/data/seed_train_full10s_5k.pkl`；数据和预训练权重不入 Git。

## 代码

| 文件 | 职责 |
| --- | --- |
| `agents/frozen_fb.py` | 严格加载 B/归一化器；safe_open 按前缀读取；FP32 累加专家 z 窗口 |
| `ppo_fb/pretrained.py` | 只构造 B、归一化器、actor、D；冻结奖励模型；现有评测适配器 |
| `ppo_fb/vec_env.py` | 专家 z 预编码、50/25/25 角色、下一帧参考、100 步切换和 timeout |
| `ppo_fb/models.py` | 原样加载的 residual actor、冻结参考 actor/归一化器、非对称多头 critic |
| `ppo_fb/storage.py` | 多头 rollout 数据，动作和 log-prob 保留裁剪前的采样值 |
| `ppo_fb/algorithm.py` | 多头 GAE、角色权重、一致性 MSE、critic 验证预热、独立更新预算与 KL 停更 |
| `ppo_fb/runner.py` | 复用 rsl-rl rollout；分阶段 rollout 长度、独立动作池刷新、定期评测、保存/恢复 |
| `ppo_fb/train.py` | 训练入口，复用原环境配置和缓存设置 |
| `ppo_fb/evaluate.py` | tracking、goal/random、留出 reward prompting、交叉回报矩阵 |
| `scripts/inspect_pretrained.py` | 参数/权重占用、专家 Σ、可选原 actor tracking 基线 |

`HumanoidVerseMjlabCore.step()` 在 `reset_idx()` 前计算 FB/D 奖励；不复制终止观测。
tracking/aux 同样用重置前的状态。goal/random 换 z 后，返回观测含新 z，切换转移按 timeout
切断 GAE；bootstrap 沿用 rsl-rl 的 V(s_t) 近似。expert 的 z 逐帧滚动，不切断 GAE。

四头顺序为 track/fb/disc/aux，权重默认为 1/0.5/0.3/0.3。权重为零的头同时停止 actor
目标与 critic loss；单头模式先加权合并奖励。参考 actor 一致性用按动作维平均的 MSE，
只作用于 goal/random。actor 的冻结 BatchNorm 不更新统计量，critic 使用独立经验归一化。

训练日志包括各角色各头奖励、各头 value/loss、验证 NRMSE/偏差/EV/回报均值/预测均值、
预热状态/覆盖率/连续达标轮数、actor/critic 更新次数与学习率、KL/停更标志、clip 比例、std 和参考策略动作距离。
tracking eval 记录 EMD 和真实刚体 MPJPE；goal/random 没有对齐动作目标，使用目标成功率、
到达时间、摔倒率和交叉回报评测，不给它们构造任意参考动作的 EMD。

tracking eval 固定使用长度 1 的 z 窗口，与 0804 历史训练 EMD 的评测一致；
训练专家 z 仍使用 checkpoint 配置的长度 8。评测后恢复评测前的训练动作池、重置环境，并刷新
rsl-rl learn 循环持有的观测对象。step/reset/eval 使用 inference mode，兼容仿真内部动态 buffer。
恢复模型、optimizer、学习率、预热状态/验证计数、动作池排列/游标/已访问专家动作与随机数状态；
物理环境重新开始，不恢复仿真快照。
恢复时沿用同一个初始 FB checkpoint 和训练选项；冻结奖励权重不一致会报错。

## critic 预热与联合训练

默认先冻结完整 π（包括 std），每轮采集 128 步，使用 λ=1 回报和 10 个 critic epoch，
不对 value 做裁剪。每个角色每 8 个环境留出 1 个环境，其完整 rollout 不参与 critic
预热梯度。验证在本轮拟合前执行，使用轨迹前半段，降低末端 bootstrap 对门槛的影响。
准备解冻的最后一轮拟合后再次复核，避免最后的更新破坏验证表现。
新一轮采集会用更新后的 critic 重新生成 bootstrap 目标。

解冻需同时满足：

- 至少 `--warmup-iterations 50` 轮；
- 每个启用的角色/奖励头：NRMSE≤`--warmup-nrmse 0.2`，平均误差绝对值≤`--warmup-bias 0.1`，
  两者均除以 max(1, 回报 RMS)；track 头只对 expert 角色设门槛；
- 有 expert 角色时，实际执行过 step 的专家动作 ID 覆盖≥`--warmup-motion-coverage 0.8`；
- 连续 `--warmup-patience 5` 轮达标。仅 EV 高或训练集 loss 低不能解冻。

达到 `--warmup-max-iterations 500` 仍不达标，会保存 checkpoint 并报错停止，π 保持冻结。
这些是固定策略价值拟合的可调验收指标，仍有采样方差与 timeout bootstrap 近似；
已有全量训练中三角色/全 tracking 分别在 198/54 轮结束预热；尚未通过多种子系统校准阈值，
不能据此保证真实 V 已完全收敛。每个启用角色至少需 2 个环境。
`--warmup-iterations 0` 显式跳过预热，可用于接口检查。

解冻后改为 24 步 rollout、λ=0.95；critic 使用 `--epochs 5`，actor 使用
`--actor-epochs 2`，均为 4 minibatch。actor 固定 lr=3e-5，critic 固定 lr=3e-4。
任一 actor minibatch 的解析 KL 超过 `--target-kl 0.01`，停止该轮后续 actor 更新；
critic 继续完成全部预算。KL 是停更条件，不是单步更新的硬上界；日志另报更新后全 rollout KL。
std 初始为 0.05，预热时冻结，解冻后可学习，范围 [0.02, 0.3]；不是固定 0.05。
value clip 在两个阶段均关闭。总 env step 预算包含预热，至多超出一个 rollout。
恢复会加载 checkpoint 的 std 和 optimizer 学习率，启动时的 `--init-std`/学习率初始化会被覆盖；
`--target-kl` 仍来自本次启动配置。固定 std 及恢复后的显式覆盖尚待实现。

训练动作池默认 `--motion-pool-size 1024` 条，与并行环境数独立，专家预处理也按此大小分批。
PPO tracking eval 同样限制每批驻留动作数，仍遍历全量动作。
动作池按随机排列逐池访问，每环境累计 `--motion-pool-refresh-steps 1024` 个控制步，
在 rollout 边界换池并重置环境、刷新基类循环持有的观测。最后一个池可少于 1024 条，
遍历完再重新洗牌。间隔必须至少容纳最长专家动作，避免定期刷新本身切掉动作末尾。
eval 不推进动作池游标，`--eval-interval 0` 也不影响换池；频繁 eval 本身仍会重置环境。

B/D 奖励和 goal 判定默认使用无噪声的物理观测，保留原缩放与冻结归一化器；actor 的观测
噪声保持原配置。`--reward-observation-noise` 启用带噪奖励消融；噪声可能同时改变非线性
B/D 奖励的均值和方差。全局 `--disable-obs-noise` 会关闭该消融所用的噪声。

## 使用

从仓库根目录运行：

```bash
# 真实环境基本检查（不做 PPO 更新）
./run_ppo_fb.sh --num-envs 32 --smoke-steps 500

# 常规训练：其余超参数使用默认值，自动启用 W&B 和 TensorBoard
./run_ppo_fb.sh \
  --init-checkpoint runs/pretrained/20260804_040010/checkpoint \
  --data-path humanoidverse/data/seed_train_full10s_5k.pkl \
  --seed 0 \
  --num-envs 2048 \
  --num-env-steps 5000000000

# M2 track-only 检查；1M 步是实验预算，尚未运行
./run_ppo_fb.sh --num-envs 64 --num-env-steps 1000000 \
  --role-fractions 1 0 0 --reward-weights 1 0 0 0 --consistency-coef 0 \
  --work-dir runs/ppo_fb_track

# 恢复时附加在同一训练命令上
# --resume runs/20261017_024105/model_10000.pt
# num-env-steps 是包含 checkpoint 已有步数的累计上限

# 参数与完整专家数据二阶矩；--max-motions 可缩小 FK 范围
.venv/bin/python scripts/inspect_pretrained.py

# 单独评测：省略 --ppo-checkpoint 即使用原始预训练 actor
.venv/bin/python -m humanoidverse.ppo_fb.evaluate \
  --mode tracking --ppo-checkpoint runs/20261017_024105/model_10000.pt \
  --init-checkpoint runs/pretrained/20260804_040010/checkpoint
```

常规启动只需指定 `init-checkpoint`、`data-path`、`seed`、`num-envs`、`num-env-steps`。
其他超参数默认采用上文的训练配置；需要调试或消融时仍可覆盖。
代码默认总预算仍是 50M，可能在预热期间用完；上面的 5B 是显式指定的长程实验预算。
运行目录按启动时间自动命名为 `runs/YYYYMMDD_HHMMSS`，W&B run 名使用同一时间戳；
TensorBoard、checkpoint 与 `config.json` 保存在该目录。默认同时开启 TensorBoard 和 W&B，
W&B project 为 `ppo-fb`，entity 从 `WANDB_USERNAME` 读取；`--no-wandb` 可仅保留本地 TensorBoard。
模型 checkpoint 仅保存在本地，不上传到 W&B；训练和评测指标仍正常记录。
关闭上传的修改需要新启动或恢复训练后生效；修改前已启动的进程仍使用旧代码，历史运行中已上传的模型也会继续保留。
保存间隔默认 1000 iter，结束训练或预热失败时额外保存 checkpoint。
默认每 1000 个 PPO iteration 评测全量动作（当前 seed 数据为 5000 条），单独评测入口也默认全量。
`--eval-interval 0` 关闭定期评测。仅在显式传入 `--eval-motions N` 时，使用 `--eval-seed 0`
从全量动作均匀、不放回抽样 N 条，供快速检查。seed 只固定子集选择，不固定仿真随机噪声。
0804 保存的模型配置 `seq_length=8` 是训练窗口；其历史评测代码写死
`end_idx = min(step + 1, ...)`，所以实际 eval 窗口为 1。PPO 的定期、独立 tracking eval
及 inspector 基线均统一使用窗口 1；动作范围为全量。比较时还需保持 DR/噪声设置一致。

消融选项：`--reward-weights`、`--consistency-coef 0`、`--single-head`、
`--no-truncate-latent-switch`、`--init-std 0.1`。白化消融先用 inspector 的
`--save-inverse PATH --shrinkage 0.1` 保存固定逆矩阵，再传 `--inverse-second-moment PATH`；
默认有界 cosine 不随 PPO 数据更新 Σ。

`evaluate --mode goal/random/prompting` 返回归一化折扣交叉回报
`R[i,j]=z_i^T mean_discounted(B(s')/d | policy_j)`。goal 成功定义为关节位置与投影重力
RMS 误差均小于 `--goal-tolerance`（默认 0.2）；摔倒定义为 pelvis 高度 <0.3 m。
`--mode prompting --tasks PATH` 使用 JSON 的 `train`/`eval` 两个列表，并拒绝重复/重叠的评测名。
每个任务指定 `name, observation, indices, target, scale`，奖励为
`exp(-mean(((observation[indices]-target)/scale)^2))`；只对 eval 任务生成 z 并报告回报矩阵。
这是可配置任务接口，尚未制定最终评测任务集。它与旧 reward inference 的物理任务和
`reward_wr_inference` 不同；同 seed 也不足以保证原始/PPO 两路抽到相同 goal/random z。
正式配对评测见 PLAN 的 P0，不能从当前 tracking 提升推断其他能力保留。

## 本次验证（2026-10-22）

- `python -m unittest discover -s tests -p 'test_ppo_fb*.py'`：61 个 CPU 用例通过，包含 checkpoint 不上传 W&B 的回归。
- 新增 PPO 源码/基础回归的 Ruff、shell 语法、lock 离线一致性检查通过。
- 重新读取三个保留运行的配置、TensorBoard 全量 eval 和关键 checkpoint 元数据；结果录入工作总结。
- 本次不重新启动 GPU 实验。以下保留早期检查的配置与限制，不代表本次再次运行。

## 早期基本验证（历史记录：2026-10-15）

- `tests/test_ppo_fb_foundation.py` 和 `tests/test_ppo_fb.py`：60 个 CPU 回归用例；
  加上 `tests/test_humanoidverse_mjlab_metrics.py` 共 80 项通过。
  覆盖真实网络加载/输出一致、冻结、BF16 累加精度、角色/参考/切换、重置前奖励、独立 GAE、
  原版单头等价、值裁剪平台、验证偏差/轨迹隔离、预热冻结/恢复/超限、KL 停更与独立更新预算、
  禁用 reward head、eval 关闭时换池、固定评测子集、预算计数、奖励噪声隔离。
- 此前全套 pytest：166 passed，2 个已有失败：README 的 `TeCH Training` 大小写，以及
  默认 LaFAN 文件未下载。新增代码 lint、语法、shell 语法与 lock 一致性检查通过。
- 真实 checkpoint：0804 actor 的确定性输出与原 FB actor 最大差值为 0。
  仅构造 35,012,706 参数；B/actor/D 权重约 0.77/121.71/11.08 MiB。
- 完整 seed 数据 FK 为 2,495,000 个控制帧；固定采样 65,536 帧测得 Σ：
  两个起点 trace≈256、最小/最大特征值≈0.1155/9.2452、条件数≈80.03。
  该分布是专家数据，不是预训练 replay；不据此宣称 Σ≈I。
- RTX 4080，32 env 各 500 步，无 NaN。0810 专家角色 track≈0.849，随机动作≈0.023；
  两次 PPO 更新及 checkpoint 恢复成功，峰值约 1.37 GiB。
- 修改前 1024 env、24 rollout steps、5 epoch、4 minibatch，两次有效 actor/critic 更新：
  49,152 env steps / 8.49 s≈5790 steps/s，峰值 2.98 GiB。
  该检查使用 seed 的 4 条动作并关闭 DR/观测噪声，不代表完整动作池的显存上界或稳态吞吐。
- 调整前的历史检查：16 env，z 窗口 8，动作取 seed 全局索引 0/1250/2500/3750，
  关闭 DR/噪声，原始 actor（以下结果不可作为当前窗口 1 的基线）：

| 起点 | 关节空间 EMD | 全局刚体 MPJPE (mm) | 去 root 平移 MPJPE (mm) |
| --- | ---: | ---: | ---: |
| 0804 | 0.7673 | 133.72 | 55.03 |
| 0810 | 0.6257 | 97.00 | 34.49 |

上述 4 动作基线只用于验证；不是全量或多种子结论。旧 `mpjpe_l` 是关节角 L2×1000，
不能解释为毫米。PPO eval 另报 `mpjpe_body_global_mm`/`mpjpe_body_local_mm` 与
`success_body_0.5m`（所有帧的去 root 平移刚体误差均 ≤0.5 m）。
定期 tracking eval 后继续训练、10 步 goal/random/任务回报接口均已在真实 GPU 环境运行。
临时动作子集、日志与 checkpoint 已清理。

本轮修改后的真实 GPU 检查（同样使用上述 4 动作，不能替代全量训练验收）：

- 16 env、128 步预热、actor 观测噪声开启：π/std 无变化，预热 checkpoint 恢复成功；
  初始 expert/track 验证 NRMSE≈1.025，严格门槛未放行。
  随后临时放宽门槛，仅验证阶段切换：转为 24 步，actor 更新 1 次后 KL 停更，
  critic 完成 4 次更新，固定 actor lr=3e-5；更新后全 rollout KL≈0.0465，证明阈值不是硬上界。
  关闭 eval 后独立换池、缓存观测刷新、干净奖励重复相等与噪声消融不同均通过。
- 默认 1024 env、128 步、1024×3 critic、10 epoch×4 minibatch，完成一次预热：
  131,072 env steps，含保存约 10.10 s，峰值 3.17 GiB；actor 未更新，checkpoint 保存成功。
  为测量该配置关闭了 DR/观测噪声，未运行到价值收敛。临时数据与输出已清理。
- 固定 1024 条 GPU 动作池，0804 checkpoint、完整 5000 条 seed 数据，开启 DR/观测噪声：
  2688 env 完成 8 轮 128 步预热、320 次 critic 更新、换池和两轮 24 步联合更新。
  PyTorch 峰值约 8.82 GiB（不含 Warp），预热后 GPU 空闲仅约 65 MiB。
  2560 env 通过两轮预热、换池和两轮联合更新，预热后空闲约 670 MiB，建议使用此档。
  2816 env 在第二轮 Warp 采样时 OOM；3072/4096 env 在首次 critic 更新时 OOM。
  联合阶段为容量检查手动解冻，不代表价值收敛；未运行此次配置的全量 tracking eval 或长期训练。
  测试没有保存训练日志、checkpoint 或探测脚本。
- 动作 FK 的矩阵转四元数改用逐元素 `where` 和 `gather`，移除两处布尔高级索引，
  保留零点的零梯度。修复后 2048 env、完整 5000 条 seed 动作完成实际入口加载、
  128 步采样和 40 次 critic 更新。此前异常间歇出现，底层触发条件尚未稳定复现。

## 仍需实验验收

已有 0804 单种子长程训练及 KL=0.02 恢复实验，结果见工作总结；
原定 M2 独立 track-only、M3 的 0804/0810 配对实验和 M4 三种子消融尚未按原方案完成。
原始 0804 的配对全量基线、正式留出任务集、能力保留、白化/余弦对照与长期资源稳定性仍需验收。

分支保留同一步 reset 的 replay 修复、全局动作索引、专家取帧和 chunked loader。
不包含 dev 的 PACE、tracking-Q critic、日志目录改造或 EMD 分组采样器；dev 分支未修改。

## FK 索引异常调查（2026-10-15）

状态：底层根因尚未查明。`where`/`gather` 是等价改写，移除了报错涉及的布尔高级索引路径，
不能据此认定原公式有错、PyTorch 存在已确认缺陷或硬件故障。

用户日志显示：0804、seed=0、2048 env，在全量专家预处理结束、恢复 1024 条训练动作池时，
原 `_sqrt_positive_part()` 报 `index 18014398509482109 ... size 300`。
该值为 `2^54 + 125`；此前另一条记录为 `2^54 + 13`。这是相同高位异常的线索，
尚无出错现场证明异常产生于坐标生成、坐标读取还是其他底层数据破坏。

本轮检查：

- 纯 PyTorch 和导入动作库依赖后的固定掩码 `nonzero()`，分别通过 177652、178909 次 NumPy 对照。
- 变动输入的原索引运算通过 59024 次；原始完整 sqrt 表达式在 `MALLOC_PERTURB_=85` 下通过 62086 次。
- CPU 上使用原始索引运算完成 30000 条真实动作 FK，未复现。
- 带 MJLab 的 2048 env 共完成 6 轮全量 5000 条动作加载，其中 3 轮保留原始表达式及临时张量生命周期。
  最后一轮直接使用 `run_ppo_fb.sh`，恢复池前十个 ID 与用户日志完全一致：
  `3663, 2195, 936, 4463, 4547, 4522, 1545, 1460, 3059, 1927`；整池加载及基本 step 均通过。
- 独立 C 整数内存校验：1 GiB、291 轮，以及跨 12 个逻辑 CPU 的 8 GiB、35 轮均通过。
  这是短时抽样检查，不能排除间歇硬件错误。
- Torch 的 libc10/libgomp/libtorch_cpu/libtorch_python、NumPy 和 Open3D 关键原生库哈希与安装 RECORD 一致。
  原字节码缓存中的两个转换函数与 HEAD 原始源码的字节码、常量和名称一致。

运行版本：Python 3.10.20、Torch 2.7.1+cu126、NumPy 1.26.4、MJLab 1.4.0、
MuJoCo/MuJoCo-Warp 3.8.1、Warp 1.14.0、Open3D 0.19.0。

诊断期间的原实现还原及异常捕获代码均已撤回，保留此前等价改写；临时文件及测试进程已清理。
尚缺下一次异常发生时的原生索引输出和 C++ 现场，现有 Python 堆栈不足以确定底层责任组件。
