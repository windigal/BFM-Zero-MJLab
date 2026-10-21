# PPO-FB 开发计划

更新：2026-10-22。已完成工作与训练曲线摘要见 [WORK_SUMMARY.md](WORK_SUMMARY.md)，
新电脑安装、数据与模型恢复见 [ppo_fb_setup.md](docs/ppo_fb_setup.md)。
下文先列当前待办，再保留最初的设计和里程碑；旧里程碑的预算不代表最新实验安排。

## 当前执行计划（2026-10-22）

### P0：建立可配对的能力评测

目的：区分冻结 B 的几何不变与 actor 使用 z 的能力变化，补齐 all-tracking 训练后的能力保留证据。

- [ ] 固定评测包：专家 motion ID、goal 帧、random z、初始 qpos/qvel、DR 参数、噪声序列、
  reward inference 用的状态池及任务定义。原始 0804 与 PPO 必须复用同一份包。
  当前只传相同 `--seed` 不够：加载 PPO 分支额外构造模型/重置，会推进随机数序列。
- [ ] 对照模型：原始 0804、`20261017_024105/model_10000.pt`、
  `20261017_170444/model_13001.pt`。检查 B 和 normalizer 的 key/shape/逐元素值，
  再对固定观测比较 B、goal z、motion z 与 reward z；相同状态上的 z 不应因 PPO 改变。
- [ ] Motion：主指标沿用全量 5000 动作、窗口 1；窗口 8 另报，不能混入同一 EMD 曲线。
  同报 EMD、真实刚体 local/global 误差和成功率；这批动作属于训练集，另建留出集才可评泛化。
- [ ] Goal：先做 64 个目标×16 个固定起点；增加持续 0.5 s 达标和最终误差，
  检查高度/速度；失败样本单独计数，不能仅比较成功样本的平均到达时间。
- [ ] Random z：64 个固定球面 z，记录摔倒率、实际 rollout 的 successor B、交叉回报矩阵、
  对角优势及行为多样性。确定性动作优先，带 std 的鲁棒性另报。
- [ ] Reward：统一任务语义和 inference 公式。现有 PPO prompting 是观测目标高斯奖励+
  `reward_inference`，旧 `reward_inference.py` 是物理任务+
  `reward_wr_inference`（softmax(10r)）；两者不能直接对比。
  0804 下载不含 replay，替代状态池必须对所有模型固定，并注明其分布。
- [ ] 至少 3 组环境随机种子；保存逐任务/逐动作结果及聚合表。

验收：评测两侧使用完全相同的输入与环境随机条件；固定输入的编码检查通过；
分别报告 motion、goal、random 和 reward 表现。旧 F 属于旧策略，不能用其 Q 作为 PPO 改善的裁判。
此项尚未实现；当前 `evaluate.py` 是接口起点，不是完整配对 benchmark。

### P1：固定 std 的最小对照

- [ ] 明确 resume 参数优先级，支持在恢复模型和 optimizer 后显式覆盖 std/学习率。
  当前 std 与 optimizer 学习率来自 checkpoint，单改 `--init-std` 不能完成实验。
- [ ] 增加固定 std 模式，实验值 0.03；冻结该参数并验证它不再进入更新。
  其他默认行为保持可复现，当前尚无此命令行开关。
- [ ] 从共同 `model_10000.pt` 分两路：原 learnable std + KL=0.01 对照；
  fixed std=0.03 + KL=0.01 实验。actor lr 先维持 3e-5。
  两路均全 tracking、disc=0，先各追加 3000 次联合更新。
- [ ] 同报 EMD/身体误差/成功率、KL、actor 更新数、clip fraction、std 和奖励，
  使用 P0 的能力保留评测；有改善再扩展到多种子和更长预算。

依据：已完成的 KL=0.02 实验增加更新量，但最终 EMD 比恢复起点高约 4.1%，
不支持仅放宽 KL。小 std 是待检验的瓶颈假设，尚不能认定为唯一原因。

### P2：拆开 disc 与角色比例的影响

- [ ] 保持全 tracking，比较 disc=0 与 disc=0.3；固定相同起点、采样预算、种子和其余参数。
- [ ] 另比较三角色与全 tracking，单独报告 goal/random/reward 能力，
  避免把专家环境数翻倍和一致性损失消失都解释成去掉 D 的收益。
- [ ] 再安排 0804/0810、cosine/固定白化、单头/多头等原计划消融，每次只改变一项。

### P3：稳定性与运行可解释性

- [ ] 原生崩溃再次发生时采集 faulthandler、core/GDB 和原生堆栈，定位换池附近的问题。
  既往 SIGSEGV 和 CPU FK 异常根因未定，不增加静默重试或掩盖错误的兜底代码。
- [ ] 明确打印退出原因及实际累计步数；预热 128 步、联合 24 步分别计算进度/ETA。
  保持总预算含预热、resume 预算是累计上限这一约定，并覆盖相应回归。
- [ ] 完整迁移本地 seed、三个关键 PPO 模型及日志，核对已入库 SHA256；
  W&B 模型上传继续关闭，不依赖其历史附件作为唯一备份。

## 原始设计与里程碑（保留作设计依据）

目标：冻结预训练 FB 表征（$B$ 与观测归一化器），在 rsl-rl 上用 PPO 重训 latent-conditioned
controller，奖励为 $r^{track}+r^{FB}+r^{D}+r^{aux}$。设计依据见 `docs/ppo_fb_controller_analysis.md`，
分支实现说明见 `docs/ppo_fb_branch.md`，实际长程结果见根目录工作总结。

已定事项：PPO 基于 rsl-rl 继承修改（与 IsaacLab/mjlab 生态一致）；0804 与 0810 两个 checkpoint 在旧机已到位；
本阶段不跑 8 卡 off-policy 对照；dev 分支不动。

## 1. 已核实的现状

### 1.1 两个预训练 checkpoint

| | 0804（`runs/pretrained/20260804_040010`） | 0810（`runs/pretrained/20260810_002844`） |
| --- | --- | --- |
| 来源 | Stage-I FB-CPR，8 卡，102M env steps | 从 0804 初始化，冻结 $B$ + tracking-Q，285M env steps |
| 含 tracking critic | 否 | 是（PPO-FB 不用） |
| $B$ | — | 与 0804 **逐元素相同**（max 差 0） |
| 观测归一化器 | — | 与 0804 近似相同（max 差 1.4e-6） |
| actor | Stage-I 策略 | tracking-Q 微调过（与 0804 差最大 6.8） |
| 判别器 | Stage-I | 继续训练过（与 0804 差最大 1.55） |

两者共有：`z_dim=256`，`seq_length=8`，$\gamma=0.98$，`actor_std=0.05`；aux 只有
`penalty_action_rate` −0.03 与 `limits_dof_pos` −10；训练数据已迁移到本仓库
`humanoidverse/data/seed_train_full10s_5k.pkl`，默认使用相对路径，Git 忽略该大文件。

这里的 `seq_length=8` 是训练窗口。0804 历史训练中的 tracking eval 写死窗口 1；
PPO 的 tracking eval 同样固定为 1，便于比较原有 EMD 日志。

$B$ 相同意味着固定输入的表征几何相同；归一化器有微小差异，actor 执行 z 的能力也不同。
可比较 **PPO 起点用 0804 actor（未经 tracking 微调）还是 0810 actor（已微调）**，
但两者判别器也不同，不能将结果仅归因于 actor 初始化。

### 1.2 需要的张量

0804/0810 的模型文件分别约 3.38/4.50 GB，PPO-FB 只需要：

| 张量 | 参数量 | 输入 | 用途 |
| --- | --- | --- | --- |
| `_backward_map` | 0.2M | `state`(64) + `privileged_state`(463) | $r^{FB}$、专家 $z$ 编码，冻结 |
| `_obs_normalizer` | — | 四个观测键的 BatchNorm 统计量 | 冻结 |
| `_actor` | 31.9M | `state, last_action, history_actor` + $z$ | PPO 初始策略 + 冻结参考 $\pi_{FB}$ |
| `_discriminator` | 2.9M | `state, privileged_state` + $z$ | $r^{D}$ |

actor 是 `ResidualActor`：残差 2048×6，均值 $\tanh(\cdot)$，无 privileged 输入，可部署。

### 1.3 rsl-rl 5.2.0 的扩展点

| 组件 | 可继承之处 | PPO-FB 需要改的 |
| --- | --- | --- |
| `VecEnv` | 抽象类：`get_observations()`、`step()` 返回 `(TensorDict obs, rewards, dones, extras)` | 写 `PpoFbVecEnv` 包装现有 mjlab env |
| `PPO` | `act`、`process_env_step`、`compute_returns`、`update`、`construct_algorithm` 都是独立方法 | 多头 value、按头 GAE、截断、一致性损失 |
| `MLPModel` | 构造签名 `(obs, obs_groups, obs_set, output_dim, ...)`，`forward(obs, stochastic_output=...)`，分布在 `self.distribution` | 写 `FbResidualActorModel`（包住 0810 的 `ResidualActor`）和多头 critic |
| `GaussianDistribution` | 支持 log-std、可学、`std_range` 截断 | 直接用 |
| `RolloutStorage` | `values/returns/advantages` 都是 `[T, N, 1]` | 继承成多头 `[T, N, K]` |
| `OnPolicyRunner` | 用 `resolve_callable(cfg["algorithm"]["class_name"])` 动态选算法 | 继承一个 runner，加 eval、z 角色、checkpoint |

`time_outs` 的 bootstrap 用的是本步 critic 值 `values`（不是终止状态的值），这是 rsl-rl/IsaacLab
一贯的近似。PPO-FB 先沿用，不额外截取终止观测（见 2.3）。

### 1.4 本机

单张 RTX 4080 16 GB，12 核，62 GB 内存。`.venv` 已装 mjlab 1.4.0、rsl-rl-lib 5.2.0、torch 2.7.1+cu126、
wandb、tensorboard。

分支：`ppo-fb` 从 `0b89378`（main）起步；实现和基本验证状态见第 6 节。

## 2. 关键设计决策

### 2.1 代码组织：继承 rsl-rl

新建 `humanoidverse/ppo_fb/`，全部通过继承实现，不改 rsl-rl 源码：

| 文件 | 类 | 继承自 | 职责 |
| --- | --- | --- | --- |
| `vec_env.py` | `PpoFbVecEnv` | `rsl_rl.env.VecEnv` | 包装 mjlab env；管理 $z$ 角色；把 $z$ 放进观测；返回多头奖励 |
| `models.py` | `FbResidualActorModel` | `rsl_rl.models.MLPModel` | 内部用 0804/0810 的 `ResidualActor`，冻结归一化器，`GaussianDistribution(log)` |
| `models.py` | `MultiHeadCriticModel` | `rsl_rl.models.MLPModel` | 非对称输入，$K$ 个输出头 |
| `storage.py` | `MultiHeadRolloutStorage` | `rsl_rl.storage.RolloutStorage` | `rewards/values/returns/advantages` 改为 `[T, N, K]` |
| `algorithm.py` | `PpoFb` | `rsl_rl.algorithms.PPO` | 按头 GAE、按头归一化 advantage、一致性损失、critic 预热 |
| `runner.py` | `PpoFbRunner` | `rsl_rl.runners.OnPolicyRunner` | 加载 checkpoint、定期 tracking eval、保存 |
| `train.py` | — | — | 入口，沿用 `run_train.sh` 的缓存目录设置 |

`OnPolicyRunner.learn()` 的采集/更新循环直接复用；子类管理预热/联合阶段长度、实际步数预算、
预热失败保存、独立换池与 eval 钩子，不复制基类循环。
`init_at_random_ep_len` 必须为 False，否则会打乱对齐的专家 rollout。

### 2.2 观测与归一化

`PpoFbVecEnv.get_observations()` 返回 `TensorDict`，分组如下：

| 组 | 内容 | actor | critic |
| --- | --- | --- | --- |
| `state` / `last_action` / `history_actor` | 与预训练一致 | ✓ | ✓ |
| `privileged_state` | 463 维 | | ✓ |
| `latent` | 当前 $z$（256） | ✓ | ✓ |
| `reference` | 下一帧参考 `state`+`privileged_state`，非 expert env 补零 | | ✓ |
| `role` | expert / goal / random 的 one-hot | | ✓ |

actor 必须用**冻结的 FB 归一化器**（预训练 actor 只认它），所以 `FbResidualActorModel.update_normalization`
为空操作。critic 用 rsl-rl 自带的 `EmpiricalNormalization`。

### 2.3 奖励在 env 内、重置前计算

环境在 `step()` 里先 reset 再取观测。aux 与 tracking 奖励本来就在重置前算。$r^{FB}$ 和 $r^{D}$ 也需要
$s_{t+1}$，所以同样放到 `HumanoidVerseMjlabCore.step()` 里、`reset_idx` 之前计算：env 持有一个
`FrozenRewardModel`（冻结的 $B$、归一化器、判别器，GPU 上 `torch.no_grad`）。这样不需要额外的
"终止观测"。time-out 的 bootstrap 沿用 rsl-rl 的做法（用 $V(s_t)$ 近似）。

### 2.4 奖励（$K=4$ 个头）

| 头 | 每步定义 | 作用的 env |
| --- | --- | --- |
| track | `tracking_rewards["total"]`，[0, 1] | expert-z（其余为 0） |
| fb | $\cos\big(B(s_{t+1}),z\big)$ | 全部 |
| disc | 判别器 logit 裁剪到 $[-5,5]$ 再除以 5 | 全部 |
| aux | $-0.03\,$action_rate $-10\,$limits_dof_pos（与 0804/0810 一致） | 全部 |

每头单独做 GAE，advantage 按头归一化后加权求和：
$A_t=\sum_k w_k\,\hat A^{(k)}_t$。初值 $w=(1.0,\ 0.5,\ 0.3,\ 0.3)$，按 env 角色可再乘系数（2.5）。
$r^{FB}$ 是否改白化，等 M0 测完 $\Sigma$ 条件数再定。

### 2.5 $z$ 的分配与 GAE 截断

| 角色 | 比例 | $z$ 来源 | 切换 |
| --- | --- | --- | --- |
| expert-z | 50% | 对齐动作，`encode_expert_latents` 逐步 $z_t$ | 随帧滚动，不截断 |
| goal-z | 25% | 专家缓冲区随机帧的 $\mathrm{proj}(B(s_g))$ | 每 100 步 |
| random-z | 25% | 球面均匀采样 | 每 100 步 |

goal/random 换 $z$ 的那一步，`extras["time_outs"]=1` 且 `dones=1`，即把切换当作 time-out：
GAE 在此截断，并按 rsl-rl 的方式 bootstrap。下一步观测里的 `latent` 已是新 $z$。

### 2.6 策略、critic 与超参（初值）

- actor：`ResidualActor` 权重原样加载；`GaussianDistribution(std_type="log", init_std=0.05,
  std_range=(0.02, 0.3))`。初始 0.05 与预训练 `actor_std` 一致；预热后 std 可学习，0.1 作为初始值消融。
  采样动作送 env 前裁到 $[-1,1]$，log-prob 用裁剪前的值。
- 冻结参考 $\pi_{FB}$：另存一份初始 actor。goal-z / random-z 样本上加
  $\beta\lVert\mu_\theta-\mu_{FB}\rVert^2$，$\beta=1.0$。
- critic：1024×3 残差 MLP，4 头；关闭 value clip，直接拟合各头原始回报。
- PPO：24 步 rollout，actor 2 epoch、critic 5 epoch，4 minibatch，policy clip 0.2，KL 停更阈值 0.01，$\gamma=0.98$，$\lambda=0.95$，
  熵系数 0，梯度裁剪 1.0。
- 学习率：actor 与 critic 分开两个参数组（rsl-rl 默认共用一个，需覆盖 `__init__`）。actor
  固定 $3\times10^{-5}$，critic 固定 $3\times10^{-4}$；KL 超标只停止本轮后续 actor 更新。
- critic 预热：冻结完整 actor（含 std），128 步 rollout、$\lambda=1$、10 个 critic epoch。
  每个角色留出约 1/8 环境整段验证，不参与预热梯度；每轮拟合前检查前半段轨迹。
  至少 50 轮，相关角色/头的 NRMSE≤0.2、归一化平均偏差≤0.1，专家实际访问动作 ID
  覆盖≥80%，连续 5 轮达标后解冻。500 轮仍未达标则保存 checkpoint 并报错停止。
  误差除以 max(1, 回报 RMS)；这些可调阈值需要长期实验校准，不保证真实价值函数已收敛。
- 训练动作池每环境 1024 个控制步在 rollout 边界刷新，与 eval 独立，随机排列遍历动作 ID。
  评测默认每 1000 个 PPO iteration 遍历全量动作；显式指定动作数时才按固定 seed 抽子集。
  奖励观测默认只缩放，噪声通过消融开关开启。
- 常规 CLI 只需输入 checkpoint、数据路径、seed、环境数、总环境步数；其余超参数使用上述默认值。
  W&B/TensorBoard 默认同时开启，保存间隔 1000 iter，运行目录自动使用 `runs/YYYYMMDD_HHMMSS`。
- env 数：最初用 1024；后续 RTX 4080 长程实验采用 2048，容量结果见工作总结。

## 3. 原始里程碑

以下保留最初拟定的验收任务；当前执行顺序、预算和未完成项以文首 P0–P3 为准。
尤其 50M 总步数可能只够 V 预热，不再用作长程联合训练的充分预算。

### M0 准备与基线（约 1 天）

- 原拟按 buffer 修复、tracking rollout、`frozen_fb` 分组交付；是否及如何提交由用户决定。
- 使用已迁移到 `humanoidverse/data/` 的 `seed_train_full10s_5k.pkl`（`.gitignore` 忽略该路径）。
- 扩展 `frozen_fb.py`：用 `safe_open` 按前缀只读 $B$、归一化器、actor、判别器。
- `scripts/inspect_pretrained.py`：
  - 打印两个 checkpoint 所需张量的参数量与显存；
  - 在专家数据上算 $\Sigma$ 的特征谱、迹与条件数 $\kappa$；
  - 写一个满足 `extract_model`/`agent.act` 接口的适配器，用现有 `HumanoidVerseMjlabTrackingEvaluation`
    分别评测 0804、0810 actor，得到 EMD / MPJPE 基线。
- 通过标准：$\mathrm{tr}\,\Sigma\approx256$；0810 的 EMD 优于 0804；两者量级与原训练日志一致。

### M1 环境与 VecEnv（约 2 天）

- `FrozenRewardModel` 与在 `step()` 内、重置前计算 $r^{FB}$/$r^{D}$（2.3）。
- env 角色、$z$ 管理、切换步的 time-out 标记（2.5）。
- `PpoFbVecEnv`：观测分组（2.2），`rewards` 为 `[N, K]`，`extras` 带 `time_outs` 与 `log`。
- 通过标准：
  - 单元测试：角色比例、切换步、$z$ 与参考帧对齐、奖励在重置步用的是重置前状态；
  - 真实 env 跑 500 步，无 NaN；expert env 上用 0810 actor 的 track 奖励明显高于随机动作。

### M2 rsl-rl 子类（约 2–3 天）

- `FbResidualActorModel`、`MultiHeadCriticModel`、`MultiHeadRolloutStorage`、`PpoFb`、`PpoFbRunner`。
- 通过标准：
  - actor 学习率为 0 时，确定性输出与 `FBModel.act(mean=True)` 完全一致；
  - 多头 GAE 单元测试（截断、time-out bootstrap、各头独立，单头时与 rsl-rl 原版结果一致）；
  - 64 env、只开 track 头跑 1M 步，track 奖励上升且 EMD 不变差；
  - 记录 env steps/s 与峰值显存，确定 env 数。

### M3 完整 PPO-FB+Track（约 2 天 + 训练时间）

- 四个头、一致性约束、三种角色全开。分别从 **0804** 与 **0810** 初始化，各跑 50M env steps。
- 日志（TensorBoard + W&B）：各头奖励与 value loss、KL、clip 比例、std、各角色 EMD、
  $r^{FB}$ 与 EMD 的背离、到 $\pi_{FB}$ 的动作距离。
- 通过标准：expert-z 的 EMD 优于各自起点的 M0 基线，goal-z 成功率不低于基线。

### M4 评测与消融（持续）

- 评测：tracking（EMD/MPJPE/成功率）、goal（成功率/到达时间）、reward prompting（训练/评测任务集分开）、
  交叉回报矩阵 $R_{ij}=z_i^\top\psi_j$、random-z 摔倒率。
- 消融（单卡、同预算、3 个种子；起点取 M3 中更好的那个）：
  1. 去掉 $r^{FB}$
  2. 去掉 $r^{D}$
  3. 去掉一致性约束
  4. $\cos(B,z)$ vs 白化
  5. 单头 vs 多头 critic
  6. $z$ 切换处截断 开 / 关
  7. 初始 std 0.05 vs 0.1

## 4. 风险与应对

| 风险 | 表现 | 应对 |
| --- | --- | --- |
| $r^{FB}$ 被 hack | $r^{FB}$ 上升而 EMD 或自然度变差 | 裁剪、降权、加判别器权重、监控背离 |
| 初始策略被冲掉 | 前几十次迭代 EMD 急剧变差 | critic 验证预热、小 actor lr、KL 停更、一致性约束 |
| 冻结判别器很快被骗过 | $r^{D}$ 饱和、动作不自然 | M3 后试按 AMP 方式继续训练判别器 |
| rsl-rl 单 lr 设计 | actor/critic 学习率无法分开 | 两个固定学习率参数组，独立更新次数；KL 停更不影响 critic |
| 单卡吞吐不够 | 50M 步超过 2 天 | 减 env 数延长训练，先跑 M3 的一个起点 |
| 显存 | 1024 env + 2048×6 actor + critic 超 16 GB | critic 缩小、增加 minibatch 数、必要时 bf16 |

## 5. 不做的事

- 不跑 8 卡 off-policy 对照。
- 不重新训练 $B$，不做跨本体。
- 不改 dev 分支。
- 不在本阶段实现 on-policy successor-feature critic（分析文档 10 节候选 3）。


## 6. 最初交付与基本检查（历史记录：2026-10-15）

完整使用方式、指标定义和已知限制见 `docs/ppo_fb_branch.md`。

- M0 基础代码：buffer 修复、tracking rollout、frozen FB，均保留为未提交修改。
  seed 数据在仓库内；选择性加载、专家 Σ 检查和原 actor 评测适配器已实现。
- M1：三角色、逐帧专家 z、下一帧参考和切换 timeout 已实现。真实 GPU 32 env×500 步无 NaN；
  0810 专家环境 track≈0.849，随机动作≈0.023。
- M2/M3 代码：rsl-rl 子类、多头 GAE/归一化、固定双学习率、验证控制的 critic 预热、KL 停更、一致性 MSE、
  保存/恢复及 TensorBoard/W&B 已接通。此前 1024 env、24 步、5 epoch 配置完成两次有效更新，
  约 5790 env steps/s、峰值约 2.98 GiB（4 动作、无 DR/噪声的短程检查）。
- M4 入口：tracking、goal/random、留出目标奖励任务、交叉回报及计划中的消融开关已实现。
  tracking 另报真实刚体 MPJPE；旧 mpjpe_l 是关节角指标，不能用作 mm。
- 专家全量 FK 2,495,000 控制帧，抽样 65,536 帧：trace≈256、κ≈80.03；两个起点一致。
  调整前窗口 8 的 4 动作检查中 0810/0804 EMD≈0.6257/0.7673；当前窗口 1 的全量基线尚未跑。
- 基础与 PPO CPU 回归共 56 个通过；全套 pytest 166 passed、2 个 main 已有失败
  （README 标题大小写、默认 LaFAN 数据缺失）。新增代码 lint 与 lock 检查通过。
- review 修复后：预热验证、KL 停更、独立换池和奖励噪声开关已做 CPU/真实 GPU 检查。
  默认 1024 env×128 步预热（10 critic epoch）运行与保存成功，峰值约 3.17 GiB；
  仅用了 4 动作且无 DR/噪声，不代表全量长期收敛。详细条件见分支文档。

当时仅交付实现和基本检测。后续已开展 0804 单种子长程训练，见 [工作总结](WORK_SUMMARY.md)；
原定 M2 的独立 1M track-only、M3 的 0804/0810 配对实验以及 M4 三种子消融仍未按原方案完成。
goal/random 没有对齐参考动作，
其质量用成功率/到达时间/摔倒率衡量；EMD 用于 tracking，不构造任意动作目标。
