# PPO-FB：冻结 FB 表征后用 PPO 重训 controller 的分析

本文记录两部分内容：

1. 对"冻结 F/B，用 PPO 最大化 $r^{FB}+r^{D}+r^{aux}\,(+\,r^{track})$"这一改造的理论与实验设计分析。
2. 跨回合 reset 问题的追踪、修复方案审查与迁移注意事项。

归档说明（2026-10-22）：本文保留开发初期的理论分析和 reset 审查。成文时第 1 部分没有跑实验，
也没有实测 $\Sigma$；第 2 部分标注“已验证”的结论来自 CPU buffer 模拟。
后续已下载 checkpoint、测量专家 $\Sigma$、加入回归并开展 GPU 训练；当前进度和实测结论以
[工作总结](../WORK_SUMMARY.md)、[实现说明](ppo_fb_branch.md) 和 [当前计划](../PLAN.md) 为准。
旧预训练 replay 仍未取得，专家分布测量不能代替 replay 分布。

## 符号

| 符号 | 含义 |
| --- | --- |
| $z\in\mathbb R^{d}$ | 行为 latent，$d=256$，投影到球面 $\lVert z\rVert=\sqrt d$ |
| $B(s)\in\mathbb R^{d}$ | backward 表征；仓库末层是 $\sqrt d$ 归一化，故 $\lVert B(s)\rVert=\sqrt d$ |
| $\Sigma=\mathbb E_{\rho}\big[B(s)B(s)^\top\big]$ | $B$ 的二阶矩；由归一化得 $\mathrm{tr}\,\Sigma=d$ |
| $\phi(s)=\Sigma^{-1}B(s)$ | 白化特征 |
| $F(s,a,z)\in\mathbb R^{d}$ | forward 表征；$Q_z(s,a)=F(s,a,z)^\top z$ |
| $r_z(s)=\phi(s)^\top z$ | latent $z$ 对应的奖励 |
| $\gamma$ | 折扣，仓库取 $0.98$ |

## 0. 结论摘要

- 方向在理论上成立，但更准确的说法是"冻结 $B$（连同 $\Sigma$），不冻结 $F$"。$F$ 是依赖 policy 的 critic，$B$ 与 $\Sigma$ 才是奖励基。
- 冻结 $B$ 后，FB 项在 $\mathrm{span}(B)$ 内与 PPO 最大化 $r_z(s')=B(s')^\top\Sigma^{-1}z$ 的回报等价。整体目标与"冻结 B 的 FB-CPR 继续训练"相同，差别在优化器、z relabeling、状态分布和 critic 外推。
- 最大风险有两个：冻结的 B 被 PPO 当奖励模型强力优化而被 hack；没有 relabeling 后，非 mocap z（random z、reward prompting）的覆盖退化。
- 最值得先做：从 $\pi_{FB}$ 初始化的 PPO-FB+Track，与"冻结 B 的 FB-CPR + tracking-Q"在同等预算下对比。

## 1. $r^{FB}$ 与 $F(s,a,z)^\top z$ 是否同一目标

z 与奖励的对偶关系：

- reward inference：$z_r=\mathbb E_{\rho}\big[r(s)B(s)\big]$。
- 反过来，落在 $\mathrm{span}(B)$ 内、对应 $z$ 的最小范数奖励是 $r_z(s)=B(s)^\top\Sigma_{\rho}^{-1}z$，其中 $\Sigma_{\rho}=\mathbb E_{\rho}[BB^\top]$。代入可验证 $\mathbb E_{\rho}[r_z B]=\mathbb E_{\rho}[BB^\top]\Sigma_{\rho}^{-1}z=z$。
- 仓库 `fb/agent.py` 的 `q_loss`（`q_loss_coef=0`，默认关闭）用的隐式奖励正是 $B(s')^\top\Sigma^{-1}z$，$\Sigma$ 取 batch 协方差。
- 奖励记在 $s_{t+1}$ 上，与 FB loss 中 $M$ 从 $s_{t+1}$ 起算一致。

综合起来，PPO 要最大化的 FB 逐步奖励是

$$
r^{FB}_t=\phi(s_{t+1})^\top z=B(s_{t+1})^\top\Sigma^{-1}z .
$$

冻结 B 时的精确结论：

- FB 的 TD 损失在 $s'$ 维度上是 $L^2(\rho)$ 意义下向 $\mathrm{span}(B)$ 的投影，Bellman 算子作用在 $(s,a)$ 维，两者可交换。
- 只要 $F$ 表达力足够，TD 不动点就满足

$$
F(s,a,z)^\top z=Q^{\pi_z}_{r_z}(s,a)=\mathbb E_{\pi_z}\Big[\sum_{k\ge 0}\gamma^{k}\,r_z(s_{t+k+1})\,\Big|\,s_t=s,\,a_t=a\Big],
$$

  对 $r_z$ 没有低秩误差（$r_z$ 本身就在 $\mathrm{span}(B)$ 里）。

加上 $D$ 与 aux 后，actor loss 对应的逐步奖励。仓库 actor loss 为

$$
\mathcal L_\pi=-Q_{fb}-0.05\,\bar w\,Q_D-0.02\,\bar w\,Q_{aux},\qquad \bar w=\mathbb E\,\lvert Q_{fb}\rvert\ \ (\texttt{scale\_reg=True}),
$$

其中 aux 奖励已除以 EMA 标准差 $\sigma_{aux}$。对应的逐步奖励为

$$
r_t=r_z(s_{t+1})+0.05\,\bar w\,r^{D}_t+0.02\,\bar w\,\frac{r^{aux}_t}{\sigma_{aux}} .
$$

$\bar w$ 随训练自适应，PPO 里的固定 $\lambda$ 只在某一时刻与它等价。PPO 中对应做法是多头 critic，按头归一化 advantage。

等价需要：相同的 $\gamma$（0.98）；$\Sigma$ 沿用预训练时的估计并冻结；$B$ 的输入预处理一致；timeout 时 bootstrap；不引入新的 termination（奖励偏移会改变最优策略）；$z$ 先验分布相同；$z$ 切换处按任务边界处理。

实际差异：

- 状态分布：off-policy 在 replay 分布上最大化 $Q$，PPO 在 $\pi_z$ 自己的访问分布上优化。容量有限时，FB 的 actor 在更宽的状态集上训练，对扰动和 $z$ 切换更鲁棒。
- critic 外推：FB 的 actor 最大化学到的 $F$，会利用 $F$ 在分布外动作上的高估；PPO 用真实回报的 GAE，对 $r_z$ 更忠实。
- relabeling：FB 每条转移可为大量 $z$ 提供信号；PPO 只服务 rollout 用的那个 $z$。
- $\Sigma$ 漂移：冻结 $B$ 的 off-policy 续训里，replay 分布 $\rho_t$ 在变，隐含奖励 $B^\top\Sigma_{\rho_t}^{-1}z$ 也在漂移（dev 现有的 Stage-II 就有这个问题）。PPO-FB 冻结 $\Sigma$，反而更平稳。

## 2. 冻结 F 的 policy-critic 失配

- $F(s,a,z)$ 是 $\pi_z$ 关于特征 $\phi=\Sigma^{-1}B$ 的 successor features，是 critic，不是表征。
- 冻结 $F$ 再训 actor：actor 收敛到 $\arg\max_a Q^{\pi_{old}}(s,a)$。如果 $\pi_{old}$ 本来就对 $F$ 贪心，只是重新蒸馏一遍旧 policy；如果额外加了 tracking、consolidation 等项，这就是一步 policy improvement，离开信任域后 $F$ 就过时了。
- 更严重的是 $F$ 不再看到新动作，actor 对 $F$ 在分布外动作上的外推误差的利用得不到纠正。长时间训练下失配很严重，只适合作短程正则。
- 改用 $r^{FB}$ 能避免这个问题：$r^{FB}$ 只依赖 $B$，与 policy 无关，$V$ 在 on-policy 数据上重新学。剩余风险转移到 $B$ 在分布外状态上的准确性。
- 代价：$F$ 不再能评估新 policy 在其他奖励上的价值，基于 $F$ 的 GPI 与 $Q$ 引导的 $z$ 选择随之失去。可在 PPO 训练后，用冻结的 $B$ 做一次纯 policy evaluation 的 FB TD，重新拟合 $F^{\pi_{PPO}}$（没有 actor，较稳定）。

## 3. 去掉 $Q_D$ / $Q_{aux}$ 会丢什么

不会丢的：价值对奖励线性，

$$
V_{total}=V_{FB}+\lambda_D V_D+\lambda_A V_{aux},
$$

"提前规避关节限位"这类长时程能力 $V$ 同样能学到。

会丢的：

- $z$ relabeling：$D$ 奖励和 FB 项在 off-policy 下能对任意 $z$ 复用数据，这是 CPR 覆盖率的核心优势。
- $(s,z)$ 不匹配样本：relabel 让 actor 见过"状态来自别的 $z$ 的 policy"这种组合。tracking inference 每步换 $z$，这种鲁棒性很重要；PPO 只见过自洽的 $(s,z)$。
- 按项分开的动作梯度：FB-CPR 每一项各自给出确定性动作梯度；PPO 混成一个标量 advantage，尺度大的项会压过其他项。
- $z$ 切换处的 GAE 边界：若 $z$ 在 episode 中途切换，跨切换点的 GAE 会把下一个任务的回报算进上一个任务的 advantage。切换点要截断，并用 $V(o_{t+1},z_{old})$ bootstrap。FB 没有这个问题，因为每条转移用自己的 $z$ bootstrap。
- 判别器奖励形式：仓库用 logit（`nn_models.py` 的 `compute_reward`），无界。PPO 下建议换 AMP 的有界形式，或对 logit 裁剪。

$$
r^{D}=\log D(s,z)-\log\big(1-D(s,z)\big)\qquad\text{（仓库，无界）}
$$

## 4. PPO 是否更适合这组奖励

- 适合的部分：dense tracking、dense aux、判别器奖励配 PPO 是成熟路线（DeepMimic、AMP、ASE、CALM、PHC、PULSE、BeyondMimic，Extreme-RGMT 也是 PPO）。on-policy 天然适应非平稳的 D 奖励，没有 Q 高估问题，大规模并行使样本效率不再是主要约束。
- 弱点在 $r^{FB}$：它本质是 256 维球面上的多任务 RL，FB 选 off-policy 正是为了 relabeling。PPO 只能靠网络对 $z$ 的泛化与大量并行 $z$ 弥补，每个 $z$ 的信号来自一条轨迹的 GAE，方差大。
- off-policy 也能大规模并行（如 FastTD3），"PPO 更好"不是先验结论。
- 判断：tracking 精度上 PPO 大概率占优；零样本覆盖上 off-policy 有结构性优势。

## 5. $r^{FB}$ 的具体风险与处理

- 尺度：仓库 $B$ 的最后一层是 $\sqrt d$ 归一化（`nn_models.py` 的 `Norm`），$z$ 也投影到半径 $\sqrt d$ 的球面上，所以

$$
B(s)^\top z=d\cos\big(B(s),z\big)\in[-256,\,256],
$$

  random $z$ 下标准差约 $\sqrt{z^\top\Sigma z}\approx 16$。$\phi^\top z$ 在 $\Sigma$ 病态时无界。建议用 $\cos(B(s'),z)$ 作 $r^{FB}$，按 $z$ 归一化尺度，多头 critic 并按头归一化 advantage。
- reward hacking（最大风险）：冻结的 $B$ 是神经网络奖励模型，PPO 会对它做对抗式优化；摔倒、扭曲、高速抖动等分布外状态上 $B$ 的值是任意的。缓解：$D$ 把状态拉回 motion 流形；aux 惩罚；$B$ 的 ensemble 分歧或到预训练状态集的 kNN 距离作分布外惩罚；对 $\pi_{FB}$ 加 KL 或一致性约束；裁剪 $r^{FB}$。同时监控 $r^{FB}$ 与不参与训练的指标（EMD、留出的判别器得分）是否背离。
- 追逐高投影状态：奖励只依赖状态，$\pi_z$ 会走到 $\arg\max_s\phi(s)^\top z$ 然后停住，random $z$ 表现为极端静态姿势。这是 FB 语义本身决定的，靠 $D$ 和 aux 约束。
- 语义塌缩：$r^D$、$r^{aux}$ 占主导时 policy 可能忽略 $z$。便宜的诊断（利用 $r_z$ 对 $z$ 线性）：对每个 $z_j$ 跑 rollout，算

$$
\psi_j=\sum_{k\ge 0}\gamma^{k}\,\phi(s_{t+1+k}),\qquad R_{ij}=z_i^\top\psi_j ,
$$

  即矩阵 $R=Z^\top\Psi$ 一次给出所有 $(i,j)$ 的交叉回报（$\pi_{z_j}$ 在奖励 $r_{z_i}$ 下的回报）；对角占优率与归一化 regret 可直接衡量语义是否保留。
- 梯度冲突：多头 advantage，统计各头 policy gradient 的余弦（类似仓库已有的 `tracking_q/*` 诊断）；不同 env 角色用不同奖励权重（见第 8 节）。
- termination 与奖励偏移：$r^{FB}$ 可为负，若有摔倒终止，policy 可能宁愿早终止。BFM-Zero 不做早停，并按 0.3 概率从摔倒姿态初始化，这是摔倒恢复能力的来源，PPO 版本应保留。

## 6. $B^\top z$ 还是 $\phi^\top z$（白化）

关键是一致性：$z_r$ 与 $r_z$ 必须互为逆映射。

- 仓库的 reward inference 是 $z_r=\sum_i r_i B(s_i)$（`fb/model.py` 的 `reward_inference`，对样本求和，没有 $\Sigma^{-1}$），与它一致的奖励是 $\phi^\top z=B^\top\Sigma^{-1}z$。
- 若改用 $B^\top z$，reward inference 要改成回归形式

$$
z_r=\Sigma^{-1}\,\mathbb E\big[r(s)B(s)\big],
$$

  即把 $r$ 对 $B$ 做最小二乘。

| | $B^\top z$ | $\phi^\top z$（白化） |
| --- | --- | --- |
| goal prompting，$z=B(g)$ | 奖励 $=d\cos(B(s),B(g))$，在 $s=g$ 处精确取最大（Norm 层保证） | 奖励 $=B(s)^\top\Sigma^{-1}B(g)$，是 Dirac 奖励的投影核，不保证在 $g$ 处最大 |
| 各方向奖励方差 | $z^\top\Sigma z$，常见行为方向占主导，random $z$ 行为多样性偏低 | $z^\top\Sigma^{-1}z$，稀有方向被放大，覆盖压力更大，但这些方向估计差、更易被 hack |
| 数值范围 | 有界 | $\Sigma$ 病态时无界 |

在这个仓库里两者差距可能不大：Norm 层使 $\mathrm{tr}\,\Sigma=d$ 恒定，正交损失在此约束下等价于压低 $\mathrm{tr}(\Sigma^2)$，会把 $\Sigma$ 推向 $I$。应先实测 $\Sigma$ 的特征谱和条件数 $\kappa$。$\kappa$ 小就用 $\cos(B,z)$（有界，goal 精确）；$\kappa$ 大就用收缩白化

$$
\Sigma_\alpha=(1-\alpha)\Sigma+\alpha I,\qquad r^{FB}_\alpha=B(s')^\top\Sigma_\alpha^{-1}z .
$$

无论哪种都要冻结 $\Sigma$，不要在 PPO 数据上重算。

```python
# 在预训练 replay 的 next_obs 上测 Σ
with torch.no_grad():
    B = model.backward_map(next_obs)                 # [N, 256]，已带 √d 归一化
    Sigma = (B.T @ B / B.shape[0]).double()
    eig = torch.linalg.eigvalsh(Sigma)
print("trace", Sigma.trace().item(), "cond", (eig.max() / eig.min()).item())
```

## 7. 原 BFM 的能力能否保留

推理只需要 $B$ 和 $\pi$，三种 prompting 都不需要 $F$，接口不变。能否保留取决于 $\pi_z$ 是否对每个 $z$ 都接近 $r_z$ 的最优。

| 能力 | 预期 | 说明 |
| --- | --- | --- |
| goal prompting | 保持或提升 | 奖励稠密，PPO 擅长 goal reaching；需要 goal-$z$ 环境 |
| tracking | 提升（加 $r^{track}$ 后） | 每步换 $z$ 的鲁棒性要专门测 |
| 插值 | 大概率保持 | 奖励对 $z$ 线性，归一化插值等于混合任务；PPO 对 $z$ 的依赖可能不如 FB 平滑 |
| 任意 $z$ 覆盖 | 退化风险 | 由 $z$ 先验与 env 比例决定 |
| reward prompting | 退化风险最大 | $z_r$ 远离 mocap 流形，训练中少见；可加 reward-$z$ 环境，但训练任务集与评测任务集必须分开 |
| 生成式 latent 模型 | 基本兼容 | $z$ 由冻结 $B$ 锚定；用旧 policy 闭环数据训练的模型（如 $z$ 空间动力学）需重新拟合 |
| 依赖 $F$ 的功能（GPI、critic 引导的 few-shot 适配） | 丢失 | 除非按第 2 节重新拟合 $F$ |
| 摔倒恢复 | 取决于设计 | 取决于是否保留"不早停 + 摔倒姿态初始化" |

## 8. 有了 $r^{track}$，$r^{FB}$ 和 $r^{D}$ 还要不要

要，但作用要分清：

- expert $z$ 上的 $r^{FB}$：$z_t=\mathrm{proj}\big(\tfrac1K\sum_{k}B(s_{t+k})\big)$，$r^{FB}$ 近似是 $B$ 空间里与参考窗口的相似度，与 tracking 冗余。价值在于把"$z$ 的含义就是 $B$ 说的那个含义"钉住。只用 PPO-Track、以 $B$ 编码的 $z$ 为条件，得到的是"以学到的运动嵌入为条件的 tracker"（类似 PULSE），非 mocap $z$ 的语义就没有定义了。
- $r^{FB}$ 的主要作用：在 goal $z$ 和 random $z$ 上做语义正则，提供对分布外 $z$ 的鲁棒性。"防止 tracker 过度依赖 mocap 流形"主要靠它。
- $r^{D}$ 的作用相反：把行为拉向 mocap 流形，在没有 tracking 奖励的 $z$ 上保证动作自然，并压制 $r^{FB}$ 被 hack。在 expert $z$ 上与 tracking 冗余；参考动作跟不上时，允许"自然地偏离"。仓库 0818 那次 `--disable-discriminator` 的训练是这个问题的现成数据点。

按 env 角色分配奖励比全部求和更合理，比例可沿用 FB-CPR 的 60/20/20：

| env 角色 | 奖励组合 |
| --- | --- |
| expert-$z$（滚动 tracking $z$） | $r^{track}+r^{D}+r^{aux}+$ 小权重 $r^{FB}$ |
| goal-$z$ | $r^{FB}+r^{D}+r^{aux}$ |
| random-$z$ / reward-$z$ | $r^{FB}+r^{aux}+$ 较小的 $r^{D}$，可选对 $\pi_{FB}$ 的 KL |

另外训练、评测、部署的 $z$ 窗口约定要统一：仓库评测用窗口 1（`evaluations/humanoidverse_mjlab.py` 中 `end_idx = min(step + 1, ...)`），训练用 `seq_length=8`。

## 9. 实验设计

给出的 4 组是对的骨架，但混淆变量太多（优化器、critic 结构、relabeling、超参、初始化、预算都不同）。

必须补的对照：

- 同目标的 off-policy 对照：冻结 B 的 FB-CPR 在同等预算下继续训练，否则提升可能只是多训了一段。
- 补成 $2\times2$：{off-policy, PPO} $\times$ {无 track, 有 track}，其中 off-policy + track 即仓库 dev 现有的"冻结 $B$ + tracking-Q"。
- 冻结 $F$、只重训 actor：验证第 2 节的失配。
- PPO 初始化：从 $\pi_{FB}$ 初始化 vs 从零训练。

消融：去掉 $D$、去掉 aux、$B^\top z$ vs $\phi^\top z$；另加收缩白化、单头 $V$ vs 多头 $V$、对先验的 KL 开/关、$z$ 切换处 GAE 截断开/关、$\gamma$ 取 0.98 vs 0.99、$D$ 冻结 vs 继续训练。

指标要覆盖 BFM 的全部能力，不能只看 tracking：

- tracking：EMD、MPJPE、成功率（含留出动作）。
- goal：成功率、到达时间、姿态误差。
- reward prompting：训练与评测任务集分开。
- latent 一致性：第 5 节的交叉回报矩阵 $R_{ij}$。
- 插值平滑性。
- 分布外 $z$：对 $z$ 加噪、random $z$ 下的摔倒率。
- 自然度：留出判别器得分、足部滑动、jerk。
- z 切换鲁棒性、摔倒恢复。
- 计算量：env steps 与墙钟时间。
- sim2sim。

每组至少 3 个种子，各组调参预算相同。

先修仓库里会削弱 off-policy 组的问题，否则对比不公平：跨回合伪转移（见第 12 节，已修）；PACE 的困难集/已掌握集划分未接入训练；评测与训练的 z 窗口不一致。

## 10. 明确判断

- 理论上是否成立：成立。FB 项在"冻结 $B$、奖励落在 $\mathrm{span}(B)$ 内"时精确等价；整体目标与冻结 $B$ 的 FB-CPR 相同，差别只在 $\bar w$ 的自适应加权。冻结 $F$ 再训 actor 在理论上不成立。
- 最大技术风险：(1) 冻结的 $B$ 作为奖励模型被强优化，在分布外状态被 hack；(2) 没有 relabeling 后，非 mocap $z$ 的覆盖与 reward prompting 退化；(3) 次要：每步换 $z$ 的鲁棒性；若引入早停会丢摔倒恢复。
- 最优先实现的版本：PPO-FB+Track 微调，每步奖励取

$$
r_t=\lambda_T\,r^{track}_t+\lambda_{FB}\cos\big(B(s_{t+1}),z\big)+\lambda_D\,r^{D}_t+\lambda_A\,r^{aux}_t ,
$$

  具体配置：
  - actor 从 $\pi_{FB}$ 初始化，log-std 设得很小，先冻结 actor 预热 critic。
  - 非对称 critic，多头 {FB, D, aux, track}，每头单独 GAE，按头归一化后加权求和。
  - $r^{FB}=\cos(B(s'),z)$，前提是先测过 $\kappa(\Sigma)$。
  - 冻结 $B$、$\Sigma$、obs normalizer；$D$ 按 AMP 方式继续训练。
  - $z$ 先验与 env 角色比例照搬 FB-CPR；$z$ 切换处截断 GAE。
  - 不早停，$\gamma=0.98$。
  - goal-$z$ 与 random-$z$ 环境加对 $\pi_{FB}$ 的一致性约束（思路同 PACE 的 consolidation）。
  - 工程量：`uv.lock` 中 mjlab 已依赖 rsl-rl-lib 5.2.0；env 里 tracking 奖励（`_compute_extreme_rgmt_tracking_rewards`）与 aux 奖励已实现，主要是接线。
- 真正有价值的贡献点：
  - 先划清边界：用 $B^\top z$ 时，PPO-FB 在数学上就是 ASE 的 vMF 技能奖励 $\kappa\,\mu(s)^\top z$，只是编码器换成了冻结的、带时间结构的 FB 编码器。若结果只是"PPO 跟踪得更好"，与 ASE、CALM、PULSE 区别不大。
  1. 能力上：拿到 tracker 级别跟踪精度的同时保住零样本 goal/reward prompting，用"跟踪精度 vs prompting 能力"的 Pareto 曲线论证。
  2. 理论上：表征与控制解耦。给出 $\mathrm{span}(B)$ 内的精确等价、权重映射与 $\Sigma$ 漂移分析，说明 $F$ 只是训练手段，任何控制学习器都能继承 FB 语义。
  3. 方法候选：on-policy 的 successor-feature critic $\psi(o,z)$。对向量奖励 $\phi(s')$ 做 GAE，令 $V(o,z)=\psi(o,z)^\top z$，在 PPO 里保留 SF 结构，找回 GPI 与交叉回报诊断。
  4. 方法候选：带悲观项的 FB 奖励，处理奖励模型被 hack 的问题。

## 11. 论文出处与本仓库的差异

- BFM-Zero：arXiv 2511.04131。Meta Motivo / FB-CPR：arXiv 2504.11054。UFO 技术报告：https://roboparty.github.io/UFO/ 。BeyondMimic：arXiv 2508.08241。TLDR：arXiv 2407.08464。
- Extreme-RGMT（arXiv 2607.20110）是基于 PPO 的运动跟踪方法，不是 FB 框架，PACE 由它提出。它的 critic 是普通非对称 PPO 价值网络，输入含参考状态。因此"在 FB-CPR 里另加 BeyondMimic 奖励的 tracking Q critic"是本仓库自己的扩展。
- 论文中 PACE 的困难集按完成率划分（每个片段跑 5 次，完成率 < 80% 为困难）；仓库改成 EMD 最高的 10%。论文的 $\lambda_{con}$ 由 acquisition 有效样本占比驱动，是 PPO 概念，在 off-policy FB 中没有直接对应量。

## 12. 跨回合 reset：追踪、修复审查与迁移

### 12.1 环境与采集的约定

- mjlab 1.4.0 的 `ManagerBasedRlEnv.step()`：`auto_reset=False` 时不做重置，返回终止时刻的观测，调用方须先 `reset(env_ids)` 再 `step()`。
- 本仓库设置 `auto_reset=False`（`train.py`），由 `HumanoidVerseMjlabCore.step()` 在同一次 step 内调用 `reset_idx`（`humanoidverse_mjlab.py`），随后 wrapper 取观测。所以 `step()` 返回的是新回合的初始观测，终止时刻的观测丢失。
- 上游 BFM-Zero（Isaac）顺序相同：`legged_robot_base.py` 的 `_post_physics_step` 先 `reset_envs_idx` 再 `_compute_observations`。先 reset 后取观测是 IsaacGym 系环境一贯的写法。
- rsl-rl 5.2.0 的 PPO 是 on-policy 逐步存 $(o_t,a_t,r_t,\mathrm{done}_t)$，$o_{t+1}$ 就是下一行的 $o_t$，timeout 时用 `values * time_outs` 做 bootstrap（`ppo.py` 的 `process_env_step`），不需要终止观测，也不会丢第一个转移。
- BFM-Zero 的 off-policy 缓冲区需要显式的 $(o_t,a_t,o_{t+1})$ 配对，问题出在这里。

### 12.2 原代码的行为（修正我之前的诊断）

早期诊断曾把问题描述为"每个回合末尾有一条跨回合假转移，约占 1/500"。这个说法对 trajectory buffer 路径不成立，已更正：这条路径上的主要问题是"丢失每条 episode 的第一个转移"，假转移只在回绕等退化情况出现。

- 原来每行存 $(o_t,\ \mathrm{done}_{t-1})$。重置步产生的行 $R$ 是新回合的 $k=0$ 观测，却带着 flag=True，被当成上一条轨迹的最后一行。采样器的 `end_point` $=L-\text{seq}-1$ 使最后一行永远不是起点，所以真实转移 $(R\to k{=}1)$ 永远采不到。
- 假转移在 trajectory buffer 路径里只出现在退化情况：长度恰为 $\text{seq}+1$ 的轨迹（例如环形缓冲区回绕后最老的残缺轨迹）。此时 `end_point` $=0$，$\lfloor u\cdot 0\rfloor=0$ 仍给出起点 0，其 next 是下一回合的 reset 观测。
- DictBuffer 路径（`use_trajectory_buffer=False`，fb 与 tech 预设都不用）里，原来的 `indexes = ~done`（`done` 是上一步的）恰好相反：丢掉第一个转移的同时，保留了重置步上的 $(o_{T-1}\to\text{reset 观测})$ 假转移。修复后的 `~new_done` 对两条路径都正确。

CPU 上用真实 buffer 类和 `find_start_stop_traj` 模拟（5000 行、500 步回合、每 782 步一次 eval）：

| | 假转移占比 | k=0 的采样权重（相对存量） |
| --- | --- | --- |
| 原布局 | 均值 8.5e-5，0.1% 的时刻出现，出现时最高 6.7%（回绕处） | 0.59×（原因推测：只有 eval 后全重置的那一回合保留了 k=0 起点） |
| 修复后 | 0 | 1.39× |

DictBuffer 路径的结论来自代码阅读，没有单独模拟。表中数字来自 CPU 模拟，k=0 权重变化的原因是推断：原布局下只有 eval 全重置后的那一回合还保留 k=0 作为起点。

### 12.3 修复内容与审查

修复涉及两处相互依赖的改动：

- `workspace.py`：删掉跨迭代携带的 `truncated`/`done`，在强制 eval 截断之后计算 `new_done = logical_or(new_terminated, new_truncated)`，trajectory 路径存 `"truncated": new_done`，DictBuffer 路径用 `indexes = ~new_done`。含义变为"这一行所采取的那一步是否结束了回合"，即重置观测成为新轨迹的第一行。
- `trajectory.py`：`get_idxs` 的 `end_point` 由 $L-\text{seq}-1$ 改为 $L-\text{seq}$，使轨迹的最后一个合法起点可被采到。

两处必须同时生效。已验证（CPU，真实 `TrajectoryDictBufferMultiDim` 与 `find_start_stop_traj`，含时间到、eval 强制截断、各 env 相位错开、环形回绕）：

- 修复后 6 组场景共 0 条非法转移，(k=0 → k=1) 可被采到；`get_full_buffer` 也无非法转移。
- 旧存储布局 + 新采样器：每组约 20 条跨回合转移（形如 (k=T−1 → 下一回合 reset 观测)）。
- 专家 buffer（1 维 `TrajectoryDictBuffer`，窗口 seq=8）：窗口不跨动作，最后一帧现在可作为 next，符合预期。
- AST 检查：循环内已没有对 `done`/`truncated` 的残留引用；代码中没有地方把存储的 `truncated` 当折扣用，语义变化不会影响折扣计算。

由审查得出、需要知道的性质：

- 修复后每个回合的"终止步"转移 $(o_{T-1}\to\text{终止观测})$ 被丢弃，因为终止观测已被 reset 观测取代。这是"先 reset 后取观测"的固有代价；目前所有终止都是 time-out（`train.py` 里其余终止都被 assert 关闭），丢掉它影响很小。eval 前的强制截断同理，每次 eval 前每个 env 丢一条真实转移。
- `terminated` 仍存上一步的值，经 `next` 读取时恰好对应本步，约定无需改；当前它恒为 False。若以后引入真实终止，需单独验证这条（没有测过 terminated 为 True 的情形）。
- tracking-Q：对齐轨迹现在从参考帧 0 开始，且 z 在 k=0 重新编码，二者对齐；最后一帧参考对应的转移同样被丢弃。
- 采样偏置（非本次引入）：先均匀选轨迹再选起点，被 eval 截短的轨迹中靠前的步被多采。在 8×512 env、每 782 步一次 eval 的模拟里，k=1..5 约 1.2–1.4×，随调度变化。

### 12.4 迁移与遗留

- 旧 checkpoint 的 buffer：存的是旧布局。配新采样器后，每条轨迹最后一个合法起点 $(k{=}T{-}1\to\text{下一回合 reset 观测})$ 变成可采的假转移，按 $L\approx 500$ 估算约占 $1/L\approx 0.2\%$，约为原布局的 20 倍，绝对值仍小。上一轮我说成"数百倍"，是夸大。续训 0810/0817/0818 前的选择：丢弃旧 buffer 重新填充；或把各 env 的 `truncated` 前移一行后再加载（未写也未测，需处理环形回绕的 `_idx` 和 eval 附近的行）。
- `get_full_buffer`（reward inference 用）没有改动，对旧布局本来就包含每条轨迹一个假转移；对新 checkpoint 则正确。
- 采样器的另一处 `sample_observation_windows` / `sample_group_observation_windows` 仍用 $L-\text{seq}-1$，会少采最后一个合法窗口，偏保守，未改。
- 没有跑单元测试：`tests/test_pace_stage2.py` 等依赖 mjlab，本机没有装。
- 上述模拟后来已固化到 `tests/test_ppo_fb_foundation.py` 的 replay/reset 回归，覆盖非法转移、
  k=0、环形回绕及只含两个完整回合的采样边界；不依赖当时 `/tmp` 中的临时脚本。

## 13. 本文未验证的部分

- 第 1–10 节没有任何实验，也没有测 $\Sigma$ 的特征谱，是否可用 $\cos(B,z)$ 取决于它。
- 第 12 节没有在 GPU 上跑完整训练，也没有在真实 mjlab 环境上跑 `step()`；env 侧行为来自读 mjlab 1.4.0 与本仓库源码。
- 论文细节来自检索摘要，FB 原始两篇论文没有逐段核对。

