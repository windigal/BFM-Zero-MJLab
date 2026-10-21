# PPO-FB：新电脑恢复指南

更新：2026-10-22。工作进度见 [WORK_SUMMARY.md](../WORK_SUMMARY.md)，后续任务见
[PLAN.md](../PLAN.md)，完整实现说明见 [ppo_fb_branch.md](ppo_fb_branch.md)。
下列命令均从仓库根目录运行，适用于 Linux + NVIDIA GPU 的 MJLab 训练路径。

## 1. 获取代码与环境

在本轮修改由用户提交并推送 `ppo-fb` 后，新电脑执行：

```bash
git clone --branch ppo-fb https://github.com/windigal/BFM-Zero-MJLab.git
cd BFM-Zero-MJLab
git status --short --branch
uv sync --frozen
```

先安装 uv 和可用的 NVIDIA 驱动。Python 由 uv 按项目的 3.10 约束管理；
按 `uv.lock` 安装，不复制旧 `.venv`、缓存或整套 Conda 环境。
旧机验证环境为 Python 3.10.20、PyTorch 2.7.1+cu126、MJLab 1.4.0、
rsl-rl-lib 5.2.0、MuJoCo/MuJoCo-Warp 3.8.1、Warp 1.14.0，显卡 RTX 4080 16 GiB。
新机器的驱动和可用环境数需实际检查：

```bash
nvidia-smi
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

仅查看总结和计划无需 GPU、数据或模型。

## 2. Git 之外的文件

| 文件 | 大小（bytes） | 获取方式 |
| --- | ---: | --- |
| `humanoidverse/data/seed_train_full10s_5k.pkl` | 757,794,275 | 从旧机复制；下述 HF 固定版本里没有完整 seed 数据。 |
| `runs/pretrained/20260804_040010/checkpoint/model/model.safetensors` | 3,384,968,272 | 从公开 HF 下载，连同模型 JSON。 |
| `runs/pretrained/20260810_002844/checkpoint/model/model.safetensors` | 4,498,119,064 | 可选，比较另一起点时下载。 |
| `runs/20261017_024105/model_10000.pt` | 582,146,117 | 从旧机复制；后续实验起点。 |
| `runs/20261017_024105/model_13000.pt` | 582,146,117 | 从旧机复制；KL=0.01 同步数参照。 |
| `runs/20261017_170444/model_13001.pt` | 582,146,885 | 从旧机复制；KL=0.02 最终模型。 |

当前模型不上传 W&B，近期 PPO 模型未配置其他远程下载来源。
Git clone 只会恢复源码、文档、测试和三次运行的
[启动配置副本](experiments)；**不会恢复上表的本地文件**。
不要用 `git add -f` 把这些大文件塞进本次提交。

### 2.1 下载公开预训练模型

来源：[FloatRIslet/bfm-zero-seed-runs](https://huggingface.co/datasets/FloatRIslet/bfm-zero-seed-runs/tree/15264d70e0b61fc7cafb9699a746a9dc72bac084)。
固定版本 `15264d70e0b61fc7cafb9699a746a9dc72bac084` 与旧机下载元数据一致。
只取 checkpoint 和启动配置，不下载历史媒体/其他运行：

```bash
.venv/bin/hf download FloatRIslet/bfm-zero-seed-runs \
  --repo-type dataset --revision 15264d70e0b61fc7cafb9699a746a9dc72bac084 \
  --include '20260804_040010/checkpoint/*' \
  --include '20260804_040010/config.*' \
  --local-dir runs/pretrained

# 需要比较 0810 时再下载
.venv/bin/hf download FloatRIslet/bfm-zero-seed-runs \
  --repo-type dataset --revision 15264d70e0b61fc7cafb9699a746a9dc72bac084 \
  --include '20260810_002844/checkpoint/*' \
  --include '20260810_002844/config.*' \
  --local-dir runs/pretrained
```

PPO 选择性加载仍需 `checkpoint/model/` 下的 `config.json`、`init_kwargs.json` 和
`model.safetensors` 三件套，不能只复制 safetensors。原始训练 config 用于核对来源；
PPO 的实际选项以本仓库入口和运行配置为准，不沿用其中旧机的绝对路径。
这些预训练下载不含原始 replay buffer。

### 2.2 从旧机复制 seed 和 PPO 结果

在新机设置实际 SSH 地址及旧仓库路径：

```bash
OLD_HOST=user@old-host
OLD_REPO=/path/to/BFM-Zero-MJLab
rsync -av --progress \
  "$OLD_HOST:$OLD_REPO/humanoidverse/data/seed_train_full10s_5k.pkl" \
  humanoidverse/data/

mkdir -p runs/20261017_024105 runs/20261017_170444
rsync -av --progress \
  "$OLD_HOST:$OLD_REPO/runs/20261017_024105/model_10000.pt" \
  "$OLD_HOST:$OLD_REPO/runs/20261017_024105/model_13000.pt" \
  "$OLD_HOST:$OLD_REPO/runs/20261017_024105/config.json" \
  runs/20261017_024105/
rsync -av --progress \
  "$OLD_HOST:$OLD_REPO/runs/20261017_170444/model_13001.pt" \
  "$OLD_HOST:$OLD_REPO/runs/20261017_170444/config.json" \
  runs/20261017_170444/
```

需要全部模型和本地历史曲线时，另复制三个完整运行目录
`runs/20261016_105629/`、`runs/20261017_024105/`、`runs/20261017_170444/`，
包括 `events.out.tfevents.*`、`config.json` 和 checkpoint；不需要复制 cache/venv。
历史汇总指标和启动配置已随文档保留，即使不复制原始日志也能阅读。

复制/下载完成后校验已有文件（0810 等可选文件缺失时跳过）：

```bash
sha256sum --ignore-missing -c docs/ppo_fb_artifacts.sha256
```

## 3. 基本检查

无需数据、checkpoint、GPU 或 pytest 的 PPO CPU 回归：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -p 'test_ppo_fb*.py'
UV_CACHE_DIR=cache/uv uv lock --check --offline
bash -n run_train.sh run_ppo_fb.sh
```

2026-10-22 为 61 个用例通过；不等同于全仓库测试。旧的 metrics 测试使用 pytest，
项目 dev 依赖当前未包含 pytest。
恢复大文件后可进行真实环境检查，不做 PPO 更新：

```bash
./run_ppo_fb.sh \
  --init-checkpoint runs/pretrained/20260804_040010/checkpoint \
  --data-path humanoidverse/data/seed_train_full10s_5k.pkl \
  --num-envs 32 --smoke-steps 500 --no-wandb
```

## 4. 日志与训练命令

PPO 默认同时开启 TensorBoard 和 W&B，project=`ppo-fb`。
rsl-rl 的 PPO writer 使用 `WANDB_USERNAME` 选择 entity；原 FB 入口的 `WANDB_ENTITY`
说明不能直接当作此处配置。认证和用户名由新电脑自行设置，不保存在仓库：

```bash
.venv/bin/wandb login
export WANDB_USERNAME=your_wandb_entity
# 不使用 W&B 时，在训练命令末尾加 --no-wandb
```

常规五参数训练（默认三角色、disc 权重 0.3）：

```bash
./run_ppo_fb.sh \
  --init-checkpoint runs/pretrained/20260804_040010/checkpoint \
  --data-path humanoidverse/data/seed_train_full10s_5k.pkl \
  --seed 0 \
  --num-envs 2048 \
  --num-env-steps 5000000000
```

复现实验 `20261017_024105`：在上述命令末尾追加
`--role-fractions 1 0 0 --reward-weights 1 0.5 0 0.3`。
全部 tracking 不等于只保留 tracking 奖励：此命令仍含 fb 和 aux。
代码默认的 50M 总步数可能在 V 预热结束前耗尽；预算包括预热，并非联合训练的步数。
2048 是旧机常用档位，不是其他显卡的保证容量。

从共同起点追加 3000 次联合更新、保持原 KL=0.01 的恢复命令：

```bash
./run_ppo_fb.sh \
  --init-checkpoint runs/pretrained/20260804_040010/checkpoint \
  --data-path humanoidverse/data/seed_train_full10s_5k.pkl \
  --seed 0 --num-envs 2048 --num-env-steps 650526720 \
  --role-fractions 1 0 0 --reward-weights 1 0.5 0 0.3 \
  --resume runs/20261017_024105/model_10000.pt
```

上述总步数是 checkpoint 的 503,070,720 加 3000×2048×24，不是再训练 650M 步。
历史 KL=0.02 实验在此命令上追加 `--target-kl 0.02`；它已运行，不建议把它作为新的默认值。

恢复会载入 actor std 和 optimizer 学习率，覆盖启动时的 `--init-std`、学习率初始化值；
不能靠 `--init-std 0.03` 将恢复模型变成固定 std=0.03。固定 std 实验尚待实现。
保持原 checkpoint、角色、奖励、网络及预热选项；物理环境会重新 reset。

输出目录自动为 `runs/YYYYMMDD_HHMMSS`，含 `config.json`、TensorBoard 和本地模型。
每 1000 iter 保存及全量 eval，最终预算退出再保存一次。
查看本地曲线：`.venv/bin/tensorboard --logdir runs`。

独立全量 tracking 评测（省略 `--ppo-checkpoint` 可测原始 0804）：

```bash
.venv/bin/python -m humanoidverse.ppo_fb.evaluate \
  --init-checkpoint runs/pretrained/20260804_040010/checkpoint \
  --data-path humanoidverse/data/seed_train_full10s_5k.pkl \
  --seed 0 --num-envs 2048 --mode tracking \
  --ppo-checkpoint runs/20261017_024105/model_10000.pt
```

## 5. 提交范围

应提交当前源码修改、新增 `humanoidverse/ppo_fb/`、CPU 回归、两个启动脚本、
`pyproject.toml`/`uv.lock`、README、根目录总结/计划及 `docs/` 中的交接文件。
`runs/`、`wandb/`、`cache/`、`.venv/` 和大 pkl 继续忽略；现有本地实验保留。
本次没有执行暂存、commit、push 或文件上传。
