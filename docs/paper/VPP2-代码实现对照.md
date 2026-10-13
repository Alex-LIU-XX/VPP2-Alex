# VPP2 代码实现对照：论文 vs 开源仓库

> 论文：[Video Prediction Policy 2: Predict Better, Act Better (arXiv:2610.10270v2)](https://arxiv.org/abs/2610.10270)
> 仓库：`roboterax/video-prediction-policy-2`（MIT，commit `72c17ac`，2026-10-12）
> 论文解读见 [`VPP2-论文解读.md`](./VPP2-论文解读.md)
>
> **本文的每一条结论都标注了代码/文档的行号证据。** 凡是我无法在代码中定位到的论文组件，
> 一律明确写成"未在代码中找到"，不做推测。

---

## 0. 结论速览

| # | 论文组件 | 代码实现状态 | 证据 |
|---|---|---|---|
| 1 | Wan2.1-I2V-14B 视频骨干 | ✅ 完整实例化 | `configs/robodojo/train_100k.yaml:207-228` |
| 2 | MoT（视频专家 + 动作专家共享注意力） | ✅ 完整实现 | `models/wan21_14b/mot.py:14` |
| 3 | 动作专家 DiT | ✅ 两套（Action2B / Action~1B） | `train_100k.yaml:229-239`、`train_action.yaml:192-200` |
| 4 | 以视频 KV cache 为条件出动作（隐式逆动力学） | ✅ 完整实现 | `mot.py:329`、`mot.py:445` |
| 5 | Attention mask（动作可见前 N 帧视频） | ✅ 完整实现 | `vpp2.py:879`、`vpp2.py:908` |
| 6 | Flow matching 训练目标（论文式 2–3） | ✅ 数学等价（σ = 1−s 记法） | `schedulers/scheduler_continuous.py:85-102` |
| 7 | 事件级继续预训练（Stage 1，49 帧 / 30k 步 / bs 1024） | ❌ **未开源**（代码 + 数据 + 权重均无） | `README.md:101-103` |
| 8 | Chunk 级后训练（Stage 2） | 🟡 仅 RoboDojo 历史条件 Video-10k 可训 | `configs/robodojo/train_video_10k.yaml` |
| 9 | 一致性蒸馏 → 单步视频生成 | ❌ **全仓库无实现** | 全库 grep `consistency/distill/teacher/ema` 无命中 |
| 10 | Stage 3 冻结视频 + LoRA 适配 | ❌ **与论文相反**：明确禁用 LoRA，改为全量微调 | `vpp2.py:39-42`、`cli.py:41-42` |
| 11 | 统一动作空间 / 坐标对齐 $A_d, B_d$ | ❌ 未开源；仅有单数据集 z-score 归一化 | `train_100k.yaml:114` |
| 12 | T 型多视角拼图 | ✅ 实现（离线合成 + 推理侧拼图） | `convert_robodojo.py:21`、`utils/conditioning.py:52-63` |
| 13 | 详细 caption 打标（任务/末端执行器/目标物/视角） | 🟡 训练侧未开源；推理侧有 Qwen 增强 | `data.py:11-13` vs `enhance_prompt.py` |
| 14 | VLM 闭环子任务规划 | ❌ 只有开环 prompt 增强 | `enhance_prompt.py`（无闭环规划器） |
| 15 | 历史帧条件（RoboDojo memory） | ✅ 已实现（论文正文未展开） | `train_100k.yaml:162-180` |
| 16 | **推理时"以预测未来为条件"** | ⚠️ **默认路径不产生也不使用预测视频** | `vpp2.py:1115-1131`（见 2.6.1） |
| 17 | "动作专家是新初始化的" | ⚠️ 实际由视频 DiT 插值初始化 | `weights.py:28-74` |
| 18 | `torch.compile` 与时延测量 | ❌ 仓库无 compile、无计时代码 | 全库检索无命中 |
| 19 | 发布自研模型权重 | ✅ 8 个 checkpoint，HF 公开 + ModelScope 需账号 | `README.md:71-99` |
| 20 | 基准评测（LIBERO-ID/Pro/OOD、RoboDojo） | ✅ 完整脚本 + 协议 | `scripts/libero/`、`scripts/robodojo/` |
| 21 | 真机 ALOHA 零样本评测 | ❌ 未开源 | — |
| 22 | 视频预测指令遵循评测（GPT 评估器 + 人工胜率） | ❌ 未开源 | — |

**一句话总结**：仓库开源的是**"视频先验 → 策略"的后半段**（动作专家、联合训练、两个基准的完整评测），
外加**蒸馏前的** Stage-1/Stage-2 视频权重；论文最核心的**大规模事件级预训练链条、
一致性蒸馏、LoRA 泛化保护、详细 caption 数据管线**都没有开源。

---

## 1. 仓库地图

```
video-prediction-policy-2/
├── README.md                     # 论文链接、权重表、三阶段概述
├── docs/
│   ├── video_prediction.md       # 零样本视频预测（Stage-1/2 权重推理）
│   ├── robodojo.md               # RoboDojo：数据转换→Video-10k→joint 100k→评测
│   ├── libero.md                 # LIBERO：Video-10k→Action-30k→三种评测
│   ├── training.md               # RoboDojo joint 训练契约（批次/LR/断点续训）
│   └── evaluation.md             # RoboDojo 评测拓扑与指标口径
├── configs/
│   ├── robodojo/{train_video_10k,train_100k}.yaml   # 全局批次 832 / 288
│   └── libero/{train_video,train_action}.yaml       # 全局批次 64 / 256
├── src/vpp2/
│   ├── models/wan21_14b/         # ★ 模型：vpp2.py(1848) mot.py(957) action_dit.py(458)
│   ├── datasets/lerobot/         # 数据：video_event_dataset / robot_video_dataset / 归一化
│   ├── trainer.py                # 训练循环（1123 行）
│   ├── video_training.py         # RoboDojo 纯视频阶段（Video-10k）
│   ├── libero/                   # LIBERO 专用 runtime / 评测 / 协议
│   ├── inference.py              # 零样本视频预测推理
│   ├── enhance_prompt.py         # Qwen3-VL 提示增强
│   └── weights.py                # checkpoint 导出 / 配对校验
├── policy/VPP2/                  # RoboDojo/XPolicyLab 部署适配器
├── scripts/{robodojo,libero}/     # 端到端 shell 入口
└── patches/                      # RoboDojo 仿真器修复补丁
```

**两套 recipe 互不通用**（`docs/libero.md:4-6`）：

| | RoboDojo | LIBERO |
|---|---|---|
| 动作专家规模 | **Action2B**（hidden 1024 / ffn 4096 / 40 层） | **≈1B**（hidden 512 / ffn 2048 / 40 层） |
| 动作维度 | 16（双臂 EE16） | 7（单臂 + 夹爪） |
| 本体感受维 | 16 | 8 |
| 视频构图 | T 型（主视图 + 2 腕部视图） | 水平拼接（头 + 腕，224×448） |
| 训练方式 | joint：视频 DiT + 动作专家**一起训** | action_only：**冻结**全部视频专家 |
| 入口 | `scripts/robodojo/*.sh` | `scripts/libero/*.sh` |

---

## 2. 逐组件对照

### 2.1 ✅ 视频骨干：Wan2.1-I2V-14B

论文第 3 节："VPP2 builds upon the pretrained Wan2.1-I2V-14B model"。
代码完全对应（`configs/robodojo/train_100k.yaml:207-228`）：

| 参数 | 值 | 与论文的关系 |
|---|---|---|
| `model_type` | `i2v` | Image-to-Video，与"以 $o_0$ 为条件"一致 |
| `dim` / `hidden_dim` | 5120 | 14B 量级 |
| `num_layers` | 40 | — |
| `num_heads` × `attn_head_dim` | 40 × 128 | — |
| `ffn_dim` | 13824 | — |
| `in_dim` / `out_dim` | 36 / 16 | I2V 的通道拼接（含条件帧） |
| `patch_size` | (1, 2, 2) | — |
| `text_dim` / `text_len` | 4096 / 128 | UMT5 文本编码 |
| `video_attention_mask_mode` | `bidirectional` | 论文 Stage 1/2 保持双向注意力 |
| `use_gradient_checkpointing` | `true` | 显存优化 |

**实际参数量**（按发布配置在 `torch.device("meta")` 上实例化统计）：

| 模块 | 参数量 | 说明 |
|---|---:|---|
| 视频专家（Wan DiT） | **16.39 B** | 论文称 14B，实测量级相符（含 patch embed / 各层 / 输出头） |
| RoboDojo 动作专家 Action2B | **2.03 B** | 名字里的 "2B" 与实际相符 |
| LIBERO 动作专家 | **0.93 B** | ✅ **这才是论文所说"0.9B action expert"** |

> 代码在初始化时会打印每个专家的参数量（`mot.py:78-82`）。
> **独立的交叉验证**：文档说发布的 RoboDojo 配对权重约 36.85 GB
> （`docs/robodojo.md:112`），而 (16.3949 + 2.0287) B × 2 bytes（bf16）
> = **36.85 GB**，两者完全吻合，说明上表的参数量统计是可靠的。

> ⚠️ **仓库里没有 `torch.compile`。** 全库检索（含 `re.compile` 误命中）未发现对模型做
> `torch.compile` 的调用；论文的 0.12 s 时延依赖该优化，见 2.6.2。

共享编码器从 Wan 目录加载：VAE、CLIP、UMT5、tokenizer（`README.md:93-96`、`docs/robodojo.md:84`）。

### 2.2 ✅ Flow matching：数学等价，符号约定相反

论文式 (2)(3)：$x_s = (1-s)\epsilon + s\,x_1$（$s=0$ 噪声、$s=1$ 数据），
目标 $v_\theta \to (x_1 - \epsilon)$。

代码（`src/vpp2/models/wan21_14b/schedulers/scheduler_continuous.py`）用的是 **noise level $\sigma$**：

```python
# L85-94  add_noise
return (1 - sigma) * original_samples + sigma * noise
# L97-102 training_target
return noise - sample          # = ε − x₁
# L125-130 step
return sample + model_output * delta   # delta = σ_{i+1} − σ_i < 0
```

- 代码的 $\sigma = 1 - s$（$\sigma=1$ 纯噪声，$\sigma=0$ 干净数据），
  推理时 `linspace(1.0, 0.0, steps+1)`（L117-119）从噪声积分到数据。
- 因此代码学习的速度场是论文 $v$ 的**相反数**（`noise - sample` vs `x₁ - ε`），
  采样步的符号也随之相反。**两者数学等价**，不是错误，但对照公式时需要换算。
- timestep shift 用 $\phi(u) = \text{shift}\cdot u / (1 + (\text{shift}-1)u)$（L38-39）。

训练时噪声采样策略可选 `shifted_uniform` / `beta` / `pure_noise`（L30）：
- RoboDojo joint：视频 `pure_noise` + train_shift 5.0，动作 train/infer shift 3.0（`train_100k.yaml:240-250`）
- LIBERO 视频：`Beta(7,1)`（`docs/libero.md:195`）
- LIBERO 动作：train/infer shift 均为 5.0（`train_action.yaml:214-218`）

### 2.3 ✅ MoT 与动作专家

**MoT 结构**（`models/wan21_14b/mot.py:14-56`）：

- 两个专家必须同时存在（`"mixtures" must include both 'video' and 'action' experts`，L24-25）；
- 强制**层数相同、head 数相同、head dim 相同**（L42-55）——这就是 MoT 的"共享注意力、各自参数"约束；
- **每个专家持有各自的 q/k/v/o 与 FFN 权重，但每一层只有一次联合注意力**：
  `MoT.forward` 把两个专家的 q/k/v **拼接**后走同一次 `_mixed_attention`，再切分回去
  （`mot.py:811-957`，核心在 L916-929；`_mixed_attention` 在 L144-156）；
- **没有专门的 video↔action 交叉注意力层**。推理时动作 query 直接 attend 到
  "缓存的视频 K/V + 动作自身的 K/V"（`mot.py:787-797`）；
- 两个专家各自还有对文本 context 的 cross-attention（`mot.py:181-187`）；
- 联合注意力做梯度检查点（`mot.py:158-166`，`mot_checkpoint_mixed_attn: true`，
  `train_100k.yaml:201`）。

**注意力掩码**（`vpp2.py:878-906 _build_mot_attention_mask`）——单个 `[Sv+Sa, Sv+Sa]` 布尔矩阵：

| 方向 | 行为 |
|---|---|
| video → video | 由 `video_attention_mask_mode` 决定；两个发布配方都是 `bidirectional`（可选 `three_frame` / `first_frame_causal` / `per_frame_causal`，`wan_video_dit.py:450-468`） |
| action → action | 全可见 |
| action → video | 只能看到**前 `action_visible_video_frames` 个视频 latent 帧**（RoboDojo 5 / LIBERO 3） |
| video → action | **永远不可见** |

发布配方中**没有任何 causal mask**。掩码的"视频 latent 帧"粒度见下：
17 帧 RGB → VAE 时间下采样 4 倍 → `(17−1)/4 + 1 = 5` 个 latent 帧
（`vpp2.py:1223`、`wan_video_vae.py:649 temporal_downsample_factor = 4`），
所以 LIBERO 的 `action_visible_video_frames: 3` 就是"看到前 3 个 latent 帧"。

**序列长度**（`wan_video_dit.py:510-512`，`tokens_per_frame = h*w` 作用在**patchify 之后**的
latent 网格上）：240×416 → VAE 8 倍空间下采样 → latent 30×52 → patch (1,2,2) →
15×26 = **390 token/latent 帧**，5 帧共 **1950 个视频 token**；加上 32 个动作 token，联合序列 1982。
（注意：不要用像素尺寸 240×416 直接除以 patch，那样会得到 24,960 的错误数量级。）

**条件注入方式**（与论文文字描述一致，但值得写清实现细节）：

| 条件 | 注入方式 | 证据 |
|---|---|---|
| 文本指令 | UMT5-XXL → 4096 维 context；`encode_prompt` 把 mask **全部置 1** | `vpp2.py:510-518`、`wan_video_text_encoder.py:220-235` |
| 初始观测图像（CLIP 分支） | CLIP（257×1280）经 MLP 投影后**前置**到文本 context，前向时按 `ctx[:, :257]` 切回 | `wan_video_clip.py:102-107,172-194`、`wan_video_dit.py:522-533,187-204` |
| 初始观测图像（VAE 分支） | **I2V 式通道拼接**（不是 latent 拼接）：4 通道 mask + 16 通道条件帧，与 16 通道噪声 latent 拼成 `in_dim=36` | `wan_video_dit.py:418-448,482-488`、`vpp2.py:558-575` |
| 历史帧（RoboDojo） | 单独 VAE 编码 `condition_video`（要求 `T % 4 == 1`）后沿 latent 时间维拼接 | `vpp2.py:766-778`、`vpp2.py:577-609` |
| 本体感受（proprio） | `nn.Linear(proprio_dim, 4096)` → 作为**一个额外 context token** 追加，且只取 `proprio[:, 0, :]`，两个专家都可见 | `vpp2.py:117-121`、`vpp2.py:520-540`、`vpp2.py:842` |

**动作专家初始化**——论文说 "the action expert is newly initialized"，**代码不完全是**：

`src/vpp2/weights.py:28-74 initialize_action()` 从视频 DiT checkpoint 里取骨干张量，
按目标形状做 **1-D 线性插值**（`weight_resize.py:13-48`），并施加
$\sqrt{d_{\text{src}}/d_{\text{tgt}}}$ 的 **α 缩放**（L52-54）；只有
`ACTION_BACKBONE_SKIP_PREFIXES` 里的张量（动作输入/输出投影、video KV 投影、
video key norm、video token encoder）保持随机初始化（L64-71 的
`projection_initialization: "random"`）。

仓库文档对此是透明的（`docs/training.md:140-141`：
"The Action backbone is interpolated with alpha scaling; Action input/output and proprio
projections are initialized fresh"），但论文正文只写了"newly initialized"。
**这不是同一个初始化策略**，也解释了为什么论文需要靠 LoRA 保护泛化，
而开源实现改用"从视频骨干插值初始化"来达到类似目的。

**两套动作专家配置**：

| | RoboDojo Action2B | LIBERO Action~1B |
|---|---|---|
| 文件 | `configs/robodojo/train_100k.yaml:229-239` | `configs/libero/train_action.yaml:192-200` |
| `action_dim` | 16 | 7 |
| `hidden_dim` | 1024 | 512 |
| `ffn_dim` | 4096 | 2048 |
| `num_layers` / `num_heads` / `attn_head_dim` | 40 / 40 / 128 | 40 / 40 / 128 |
| `proprio_dim` | 16 | 8 |

论文写"0.9B-parameter DiT"——对应的是 **LIBERO 那套 ≈1B**（`docs/libero.md:5-6` 明确说明
"LIBERO Action expert is approximately 1B (hidden 512, FFN 2048), while the RoboDojo recipe uses Action2B"）。
所以论文的 0.9B 与 RoboDojo 的 Action2B 不是同一个东西，**读表时不要混淆**。

**隐式逆动力学（KV cache 条件）** ✅：

- `mot.py:329 prefill_video_cache(...)` —— 先跑一次视频专家，缓存 K/V；
- `mot.py:445 forward_action_with_video_cache(...)` —— 动作专家以缓存为条件去噪；
- `action_visible_video_frames`（`train_100k.yaml:203` = 5；`train_action.yaml:167` = 3）
  控制动作 token 能看到前几个视频 latent；
- 掩码构造在 `vpp2.py:879 _build_mot_attention_mask` 与 `vpp2.py:908
  _build_action_video_conditioning_mask`；
- 另有"干净首帧"路由 `action_clean_first_frame_*`（`vpp2.py:84-85`、`mot.py:455-512`）。
- 部署侧同样：`docs/robodojo.md:9-10` "inference prefills the Video cache once and denoises Action"。

> 注意：**推理阶段确实是"预填充视频缓存 → 只去噪动作"**，与论文"以视频模型的 KV cache 为条件"
> 严格一致。但**训练阶段**的差异见 2.5。

### 2.4 ❌ 事件级继续预训练（Stage 1）——未开源

论文 Stage 1 的关键数字：**49 帧 / 416×240 / 30,000 步 / batch 1024 / lr 1e-5**，
数据构造是**在整个事件上均匀采样 N 帧**（论文式 1）。

仓库中**没有任何**对应的训练配置或数据处理代码：

- 没有 49 帧事件的训练 YAML（`configs/` 下只有 `libero/` 与 `robodojo/`；
  49 只出现在**推理默认值** `src/vpp2/inference.py:194 --num-frames default=49`
  和发布的 Stage-1 权重说明里）；
- **没有论文式 (1) 的采样逻辑**。`VideoEventDataset` 名字里有 "event"，
  但它的采样是**固定步长的连续窗口**，不是事件级均匀采样：
  `obs_offsets = np.arange(num_frames) * global_sample_stride`
  （`video_event_dataset.py:532`，`global_sample_stride: 1`）解码连续帧，
  再用 `range(0, num_frames, action_video_freq_ratio)` 抽成 25 帧
  （`:537-540`）。即"97 个连续原生帧里每 4 帧取 1 帧"，
  **不是** $o_0, o_{\lfloor T/N \rfloor}, \dots, o_T$。类名与 docstring 是遗留物；
- 没有论文 Table 1/5 中任何大规模数据源（OXE/DROID/Ego4D/OpenVid…）的下载或转换脚本；
- 没有两级 caption 打标管线，也没有 1–12 s 片段的 VLM 语义分割
  （`critical_phase_sampling.py:66-119 detect_gripper_events` 只是 joint 阶段的
  **起始点采样权重**，不是事件分割）；
- 没有统一动作空间的对齐实现。

仓库自己承认了这一点（`README.md:101-103`，HTML 注释中的 TODO）：

```html
<!-- ## TODO
- [ ] Release large-scale, event-level video pretraining code and configurations. -->
```

更强的一条证据来自 **git 历史**：commit `e7e1a2a`（"chore: remove internal agent
instructions from public repository"）删除了一个内部 `AGENTS.md`，其中写道：

> "The robot-video pretrained backbone checkpoint remains a TODO and **must not be
> published until explicitly authorized**. Keep intermediate experimental stages
> out of the workflow."

（该文件已被作者主动从公开版本移除，此处仅作为**发布范围的史料**引用；
`git show e7e1a2a^:AGENTS.md` 可复现。）

**放出的是产物而非过程**：Stage-1（49 帧）与 Stage-2（17 帧）视频权重可下载
（`README.md:18`、`README.md:79-80`），可用 `scripts/infer_video.sh` 做零样本预测，
但无法从零复现这两次预训练。

> **补充语境**：`THIRD_PARTY_NOTICES.md:3-6` 说明本仓库是**从内部 FastWAM 工程抽取**的：
> "The extracted FastWAM project code … **limits the exposed training/inference modes**,
> makes data/checkpoints portable, and **removes unrelated experimental entrypoints**."
> 这解释了代码中残留的**失效分支**（例如 `vpp2.py:677` 的
> `use_action_only_fast_path = False and (...)`、`vpp2.py:1246` 的 `"joint" != "joint"`、
> `trainer.py:515` 的 `"joint" == "alternating"`）——它们是抽取时被停用而非重新实现。

### 2.5 ❌ Stage 3 的 LoRA / 冻结策略——与论文相反

**论文**（3.2 节 Stage 3）：

> "we freeze the base parameters of the video DiT during the early stages of action training and
> adapt the video backbone using only **LoRA** … This strategy helps preserve the pretrained video
> representations while limiting disruption from action-training gradients."

**代码**明确拒绝这条路：

```python
# src/vpp2/models/wan21_14b/vpp2.py:39-42
def _normalize_video_lora_config(config):
    if config and config.get("enabled", False):
        raise ValueError("This release trains the complete Video DiT; LoRA is unsupported")
    return {"enabled": False}

# src/vpp2/cli.py:41-42
if m.get("joint_denoising", False) or m.video_lora.enabled:
    raise ValueError("Joint denoising and LoRA are not part of this release")
```

配置里写着 `video_lora: enabled: false`（`configs/robodojo/train_100k.yaml:204-205`），
一旦置为 `true` 直接抛异常。

**实际做法**分两套，都不是 LoRA：

| 阶段 | 实际训练参数 | 证据 |
|---|---|---|
| RoboDojo joint 0–100k | `model.eval().requires_grad_(False)` 后 `model.dit.train().requires_grad_(True)` + 动作专家 + proprio → **视频 DiT 全量微调** | `trainer.py:479-492`、`trainer.py:494-504` |
| LIBERO Action-30k | "Frozen video K/V cache requires **every video-expert parameter to have requires_grad=False**" → **完全冻结视频专家**，只训动作专家 + proprio | `libero/action_model.py:357-361`、`docs/libero.md:228-229` |

LIBERO 的 LoRA 代码路径是**死代码**（`libero/action_model.py:325`：
`lora_graph_requested = allow_trainable_video and bool(False)`）。

**含义**：论文把"LoRA 保护泛化"当成第 1 节根因 2 的解法，而开源实现里这条策略不存在。
用本仓库复现出的 RoboDojo 策略，其泛化行为**未必**等于论文所描述的机制。
这既是复现风险，也是"论文机制未被开源代码验证"的直接证据。

### 2.6 ❌ 一致性蒸馏→单步视频生成

论文 Stage 2 用一致性蒸馏（式 5）把视频模型压成**单步**生成（推理 $F_\theta(\epsilon, 0, c)$）。

全仓库检索 `consistency` / `distill` / `teacher` / `ema` / `stopgrad` 等关键词，
**没有任何蒸馏实现**（唯一命中是 `trainer.py:303` 里无关的 "rank consistency"）。
没有 EMA 学生、没有冻结教师 flow、没有 $L_{\text{CD}}$、没有任何一致性/LCM/DMD 采样器。

仓库的做法是**直接绕开**：

- 放出的两个视频 checkpoint 都是**蒸馏前**版本（`README.md:32`：
  "we start post-training from the **Stage-2 video model before distillation**"）；
- 零样本视频推理默认 **30 步 + CFG 4**（`docs/video_prediction.md:64-66`、
  `src/vpp2/inference.py:195-197`），而不是 1 步；
- 训练侧用**纯噪声端点**替代蒸馏：`train_sampling_strategy: pure_noise`
  （`train_100k.yaml:241`、`libero/train_action.yaml:204`）在
  `scheduler_continuous.py:65-67` 里把 $\sigma$ 直接设为 1
  （注释原文："sigma=1 makes add_noise return the noise exactly, with no clean sample mixed in"），
  推理时只做一次前向。

#### 2.6.1 ⚠️ 默认推理路径**既不产生也不使用**预测视频

这是本次对照里最重要的一条发现，需要精确表述。

代码在 `src/vpp2/models/wan21_14b/vpp2.py:1606-1611` 把视频 timestep 设为
`num_train_timesteps`（=1000，即 $\sigma=1$ 纯噪声）后构建冻结视频缓存：

```python
timestep_video = torch.full(
    (latents_video.shape[0]),
    float(self.train_video_scheduler.num_train_timesteps),   # = 1000 → 纯噪声
    ...)
video_cache_result = self._build_frozen_video_cache(...)
```

而 `_build_frozen_video_cache` 在**不要求返回预测**时**根本不计算最终输出投影**：

```python
# vpp2.py:1115-1131
if return_video_prediction:
    (video_kv_cache, final_video_tokens) = self.mot.prefill_video_cache(
        **prefill_kwargs, return_final_tokens=True)
    video_prediction = self.video_expert.post_dit(final_video_tokens, video_pre)
else:
    video_kv_cache = self.mot.prefill_video_cache(
        **prefill_kwargs, max_layers=max_video_layers)
    video_prediction = None          # ← 预测 latent 压根没算
```

是否计算由 `capture_video_prediction = return_one_step_video or return_full_video`
决定（`vpp2.py:1612`），而策略侧的开关是：

```python
# src/vpp2/policy.py:398
capture_visual_plan = self.visual_planning_output_root is not None
```

`visual_planning_output_root` **默认是 `None`**（`policy.py:232`），`policy/VPP2/deploy.yml`
也没有设置它。而且即使打开了，单步视频 latent 也是在**动作去噪之后**才算出来用于可视化：
`vpp2.py:1699` 是动作采样步，`vpp2.py:1732` 才是视频 step，结果写进
`result["video_one_step"]`（L1786）并标注来源 `"action_context_cache_forward"`（L1789）。

**结论（精确版）**：
开源实现的推理路径把视频专家当作**指令/图像条件化的特征提取器**——
动作专家 attend 的是视频专家在 $\sigma=1$（纯噪声输入）那一次前向的**逐层 K/V**，
而不是任何被去噪出来的未来画面。论文所说"conditioned on the predicted future"
在开源代码里对应的是"conditioned on the video expert's single noise-timestep forward"，
**预测出的未来 latent 在默认配置下不会被物化**。这与论文式 (5) 之后那句
"At inference, the video latent is generated in a single forward pass as $F_\theta(\epsilon,0,c)$"
的读法存在解释空间。

LIBERO 侧更彻底，`docs/libero.md:178-180` 明确写着：

> *"Stage 2 encodes only the first RGB frame through VAE/CLIP, then builds all five video
> latent frames as **pure noise at timestep 1000** and caches frozen video K/V."*

并且 LIBERO 评测显式关闭了视频保存与一步视频
（`libero/evaluate.py:110 save_one_step_video=False`、
`libero/eval_libero_single.py:174` 把 `save_one_step_video` / `visualize_future_video`
列入禁止项、`:301 capture_one_step_video=False`）。

#### 2.6.2 ❌ 时延主张无法从代码验证

论文的 **0.12 s 视频 + 0.1 s 动作 = 0.22 s** 三个数字在仓库中**没有任何对应实现或测量**：

| 论文要素 | 代码情况 |
|---|---|
| bf16 | ✅ 默认 bf16（`train_100k.yaml:28`、`policy.py:43-51`） |
| `torch.compile` | ❌ **全库无 `torch.compile` 调用** |
| 视频单步生成 | 🟡 有单次前向的缓存，但默认不算输出（见 2.6.1） |
| 动作 **5** 步去噪 | ❌ 所有发布入口都是 **10** 步（`robodojo.py:8`、`deploy.yml:21-22`、`libero/evaluate.py`） |
| 时延测量代码 | ❌ 只有训练 profiler（`utils/training_profiler.py`）与评测墙钟，无模型时延指标 |

所以 0.22 s 这个数字**只能靠论文自己声明**，无法用本仓库复现或证伪。

### 2.7 ❌ 统一动作空间与坐标对齐

论文式 (6)：$\text{}^{e}T_d = A_d T_d B_d$（workspace alignment + end-effector alignment）。

代码中**找不到** $A_d$/$B_d$、`workspace alignment`、`SE(3) frame transform` 等任何实现。
仓库实际做的是**单数据集内**的归一化：

| 数据集 | 归一化模式 | 统计来源 | 证据 |
|---|---|---|---|
| RoboDojo | **z-score**（`global_mean/global_std`），16 通道含夹爪 | `configs/robodojo/normalization_ee16.json`（`num_episodes 3500`、`num_transition 1859602`），由 `data.py:121` 复制为 `dataset_stats.json` | `train_100k.yaml:114`、`docs/training.md:13-14`、`docs/robodojo.md:158-159` |
| LIBERO | **min/max**（`global_min/global_max`），含夹爪 | `configs/libero/dataset_stats.json`（`num_episodes 1712`、`num_transition 277713`） | `libero/train_action.yaml:135`、`docs/libero.md:152-154` |

归一化实现细节（`datasets/lerobot/utils/normalizer.py`）：

- 支持 `min/max` / `q01/q99` / `z-score` 三种模式（L17），但**没有任何发布配置选用 `q01/q99`**；
- `stepwise_*` 统计只在 `use_stepwise_action_norm` 为真时使用（所有配置都是 `false`），state 恒用 `global_*`（L36-69）；
- 变换是仿射并**截断到 [−5, 5]**（L150-153），`std_reg = 1e-8`、`range_tol = 1e-4`，退化维度映射到 ±1（L113-114,137-142）；
- 只有 16 通道的 z-score / min-max，**没有任何跨数据集、跨本体的坐标系变换**。

**旁证：坐标变换代码存在但是死代码。**
`datasets/lerobot/transforms/rotation.py:26 PoseRotationTransform` 是一个通用的
姿态表示转换器（四元数 ↔ 轴角 ↔ 6D/9D ↔ 欧拉），但全库检索只命中它自己的定义，
且**所有发布配置都写 `action_state_transforms: null`**
（`train_100k.yaml:112`、`libero/train_action.yaml:133`）。
`action_state_merger.py:7 ConcatLeftAlign` 也只做最后一维拼接与补零，没有坐标运算。

**EE16 的实际含义与校验**：16 维双臂**绝对**末端执行器状态/动作，
`[xyz, qwxyz, gripper] × 2`（`convert_robodojo.py:22-23` 的
`EE_NAMES = ["l_x",…,"l_g","r_x",…,"r_g"]`）。转换器会硬校验
`codebase_version=="v3.0"`、`fps==25`、`total_episodes==3500`、`total_frames==1856102`、
float32 且 shape `[16]` 且名字序列完全一致，否则报
`"expected absolute EE16 [xyz, qwxyz, gripper] x 2"`（`convert_robodojo.py:36-46`）。
数值**原样保留**，不做四元数重排、不做额外 action shift（`docs/training.md:46-49`），
夹爪是**连续**通道、不做阈值化（`docs/evaluation.md:58-59`）。

> 也就是说，开源部分只覆盖论文 80% 数据（第一视角双臂）里的**单一来源**，
> 跨数据集统一动作空间这一"提升泛化的关键设计"没有被开源。

### 2.8 ✅ T 型多视角拼图

- **离线合成**：RoboDojo 转换器把 3 路相机（`cam_high` / `cam_left_wrist` / `cam_right_wrist`，
  `convert_robodojo.py:21`）用 ffmpeg 合成 `880×480` 的 T 型 MP4——
  头部视图 `scale=640:480`，两个腕部视图先做 4:3 中心裁剪再 `scale=240:240`，
  然后 `vstack` 成右侧再与主视图 `hstack`（`convert_robodojo.py:126-133`），
  产物尺寸/帧率/帧数被硬校验为 `(880, 480, 25, length)`（`:143-147`）。
  训练时再 **直接 resize（各向异性，非裁剪）** 到 `240×416`
  （`train_100k.yaml:78-81`，`video_resize_mode: direct`；`docs/training.md:53-57`）。
- **推理侧拼图**：`src/vpp2/utils/conditioning.py:62-71` 实现单张图合成——
  主视图中心裁剪到 4:3 后缩放到 320×240 放 (0,0)，腕部视图缩放到 120×120 放 (320,0) / (320,120)，
  **黑底画布 440×240**（`canvas = Image.new("RGB", (440, 240), "black")`），
  缺失视图保持纯黑，最后 bicubic 到 416×240，并回报 `missing_views`（`:74-80`）。
  完整说明见 `docs/video_prediction.md:121-136`。
- ⚠️ **两条路径的画布不同**：训练是 `880×480 → 416×240`，推理是 `440×240 → 416×240`。
- ⚠️ **训练数据里没有黑面板**：RoboDojo 转换器要求三路相机齐全（`convert_robodojo.py:74-85`），
  所以"缺失视图填黑"只出现在推理侧与 `vpp2_processor.py:246-253` 的零张量补齐路径中。
- ⚠️ **LIBERO 不用 T 型**：改为头部 + 腕部**水平拼接**成 `[3,17,224,448]`
  （`robot_video_dataset.py:216-217`、`docs/libero.md:171-173`），
  且 `libero/cli.py:12-21` 硬校验 `([224,448], 65, 32, 4, "horizontal")`。
  所以"T 型多视角"只覆盖 RoboDojo 与零样本视频预测两条路径。

### 2.9 🟡 详细 Caption：训练侧未开源，推理侧有 Prompt 增强

**论文**要求 caption 含四要素（详细任务描述 / 显式末端执行器 / 无歧义目标物 / 相机视角）。

**开源训练侧**用的是**极简模板**（`src/vpp2/data.py:11-13`）：

```python
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)
```

RoboDojo 的 `training_prompt` 就是用它套上 `source_tasks.csv` 里的**一句话任务**
（`convert_robodojo.py:183`、`docs/training.md:27-28`），例如
*"Build a tower using the wooden blocks and wooden boards."*

此外 `VideoEventDataset` 的 `instruction_column` 默认值仍是 `"gemini_caption"`
（`datasets/lerobot/video_event_dataset.py:97`）——这是内部用 VLM 打标留下的**字段名遗迹**，
但生成 caption 的管线本身没有随仓库发布。训练侧 `training_prompt` **必须由用户提供**
（`src/vpp2/data.py:35-45` 会校验该列存在，`:176` 注释 "training_prompt already contains
the complete template"）。论文 caption 模板的四要素（任务描述 / 末端执行器 / 目标物 / 视角）
在代码中**没有对应实现**，只有推理增强器输出的 JSON 字段
（`task_goal` / `arm_choice` / `left_action` / `right_action`，`enhance_prompt.py:23-70,95-135`）。

**推理侧**倒是有一个完整的 prompt 增强器：`src/vpp2/enhance_prompt.py`
（Qwen3-VL 驱动，`QwenEnhancer` 类在 L144-193，默认 `weights/Qwen3-VL-8B-Instruct`，L276），
system prompt 里塞了大量操作常识约束（保留任务/目标/最终状态、说明左右手顺序、
结尾追加相机描述模板等，L23-70），并带一轮**任务保真度复核**
（`REVIEW_PROMPT` L72-93）；配套 `scripts/enhance_prompt.sh`。
`docs/video_prediction.md:175-178` 特别加粗强调
**"Prompt enhancement is critical and can significantly affect model performance."**

> ⚠️ **但这个增强器只接在"开环视频预测"上，没有接进策略。**
> RoboDojo 策略推理用的是同一个简单模板：
> `src/vpp2/policy.py:12` 从 `.data` 导入 `DEFAULT_PROMPT`，并在 `:400` 处
> `"prompt": DEFAULT_PROMPT.format(task=instruction)`。
> 也就是说，**基准评测的策略从未见过详细 caption**。

**含义**：论文把"详细 caption"当作泛化的主要来源（见 Table 2 的 Fixed-Step 消融论证），
但开源 recipe 训练时**只有简单指令**。因此本仓库训出的策略**不太可能**复现论文由 caption 带来的增益；
推理侧的 Qwen 增强只是**推理期**的补丁，不改变训练分布，而且还没接到策略上。

### 2.10 ❌ VLM 闭环子任务规划

论文 3.3 节与 Figure 9 描述：VLM 结合视觉观测与执行反馈，**闭环**选择下一个子任务。

仓库中**没有**闭环规划器。只有：

- 开环的 `enhance_prompt.py`（把一句话指令扩写成详细指令，一次调用，无反馈回路）；
- RoboDojo 适配器里 `default_instruction` 的兜底（`policy/VPP2/model.py:56-60`）。

因此论文 Figure 9（27.6% → 57.6%）**无法用本仓库复现**。

### 2.11 ✅ 历史帧条件（开源仓库相对论文的**增量**）

这是仓库**超出论文正文**的部分。论文 Table 7/8 显示 VPP2 在 RoboDojo 的
**Memory 维度只有 4.61 / 4.08**，远低于 GPT-6-Astra 的 43.04 / 38.67。
README 明确说明为支持需要记忆的任务，给模型输入**加入历史帧**，并且为此
**从蒸馏前的 Stage-2 视频模型开始后训练**（`README.md:32`）。

实现细节（`configs/robodojo/train_video_10k.yaml:95-113`、`train_100k.yaml:162-180`）：

| 参数 | 值 | 含义 |
|---|---|---|
| `condition_history_frames` | 8 | 8 帧历史观测 |
| `condition_history_stride` | 25 | 历史帧间隔 25 个原生帧（25 Hz 下 ≈1 s/帧，覆盖约 8 s 历史） |
| `condition_history_stride_choices` | 21/23/**25**/27/29，概率 0.07/0.07/**0.72**/0.07/0.07 | 训练期抖动，推理固定 25 |
| `condition_history_jitter_probability` | 0.1（最大 1 原生帧） | 时间鲁棒性 |
| `condition_include_episode_first` | true | 首次观测始终在条件里 |
| `condition_history_target_prefix` | true | 历史前缀参与视频损失 |

窗口构成（`docs/robodojo.md:217-223`，`cli.py:81-87` 的 dry-run 摘要）：

| 阶段 | `num_frames`（原生） | 逻辑 RGB 帧 | 条件帧 | 未来帧 | 未来时间跨度 @25 Hz |
|---|---:|---:|---:|---:|---|
| Video-10k（`train_video_10k.yaml:67`） | 97 | 25（stride 4） | 9 | **16** | 64 原生步 = **2.56 s** |
| joint 100k（`train_100k.yaml:74`） | 65 | 17（stride 4） | 9 | **8** | 32 原生步 = **1.28 s** |

校验逻辑在 `src/vpp2/video_training.py:15`（强制 `(97, 4, 8, 25)` 契约）。

**关键机制 `condition_history_target_prefix`**（`train_100k.yaml:180`）：为了让 VAE 的条件 latent 成为
目标 latent 的**精确因果前缀**，代码把目标改写成
`cat([condition_video, video[:, 1:1+future_frames]])`，其中
`future_frames = target_frames - condition_frames`（`video_event_dataset.py:895-901`）。
joint 阶段因此只保留 **8 个未来帧（stride 4 → 32 原生步 = 1.28 s）**，
**恰好等于 `action_horizon: 32`**（`train_100k.yaml:75`）——视频预测窗口与动作块在时间上严格对齐。
干净前缀还额外获得 `condition_latent_loss_weight: 1.0` 的损失权重
（`train_video_10k.yaml:125`、`wan21.py:462-473`）。

> ⚠️ **与论文的时间尺度对不上**：论文 Stage 2 说机器人 chunk 是 **8 秒**、人手 **2 秒**；
> 开源 recipe 实际是 Video-10k 的 **2.56 s** 与 joint 的 **1.28 s**，
> 而且**代码里完全没有** $H$（秒）、$f$（fps）、$\delta = fH/N$ 这套换算
> （仅 RoboDojo 断言 25 Hz：`data.py:48-49`、`critical_phase_dataset.py:42-43`）。
> LIBERO 的帧率**没有被固定**，而是从各数据集外部 `meta/info.json` 读取
> （`base_lerobot_dataset.py:136-138`），所以 LIBERO 的"每块几秒"无法从本仓库推算。
> 说明开源 recipe 是**为 RoboDojo 重新设计的**，而非论文 Stage 2 的直译。

### 2.12 ✅ 训练模式与损失

| 项目 | 值 | 证据 |
|---|---|---|
| `train_mode` | `joint`（RoboDojo）/ `action_only`（LIBERO 动作阶段）/ `video_only`（视频阶段） | `train_100k.yaml:22`、`train_action.yaml:13`、`cli.py:26-32` |
| 损失 | `lambda_video: 1.0` + `lambda_action: 1.0`，视频用 `F.mse_loss` | `train_100k.yaml:251-253`、`vpp2.py:958`、`vpp2.py:1378` |
| 精度 | bf16 | `train_100k.yaml:28`、`train_action.yaml:20` |
| 分布式 | DeepSpeed **ZeRO-1**（RoboDojo）/ **ZeRO-2**（LIBERO） | `configs/robodojo/deepspeed_zero1.json`、`configs/libero/deepspeed_zero2.json` |
| 优化器 | AdamW，grad clip 1.0，weight decay 0.01 | `train_100k.yaml:23-31` |
| 仅支持全量 joint | `cli.py:33-35` 拒绝非 full joint（除 video_only 外） | `cli.py:33-35` |
| 架构硬约束 | Action2B 必须是 `(hidden 1024, ffn 4096, layers 40)` | `cli.py:36-40` |
| 契约硬约束 | horizon 32 / history 8 / stride 25 | `cli.py:43-44` |

---

## 3. 训练资源对照

### 3.1 论文披露情况

> **论文正文与附录完全没有提及 GPU 型号、数量、节点数、训练时长或总算力。**
> 全文检索 `GPU` / `A100` / `H100` / `cluster` / `compute budget` 均无命中。
> 唯一可用的算力线索是 Stage 1 的 **30,000 步 × batch 1,024**。

**仓库同样没有硬件信息，而且这是有意为之。** git 历史里被删除的内部 `AGENTS.md`
（commit `e7e1a2a`）明确要求：

> "**Describe training scale using global batch size and training steps in public docs.**"
> （用全局批次与训练步数来描述训练规模，而不是硬件。）

所以 `docs/training.md` 只给"全局批次 288 / 100,000 步"，`docs/robodojo.md` 只给
"全局批次 832 / 10,000 步"这类**可移植的规模口径**，而非"多少张卡跑多久"。
这不是遗漏，是发布策略；对复现者的直接后果是：**必须自己按显存试算并行度**。

### 3.2 仓库可反推的参考拓扑

启动脚本的默认值 + `reference_global_batch_size` 校验可以**精确反推**参考并发度
（`src/vpp2/cli.py:70,105-107` 会核对全局批次）：

| 训练阶段 | 脚本默认并行度 | 单卡 batch | 全局 batch | 校验依据 |
|---|---|---|---|---|
| RoboDojo Video-10k | `NNODES=4` × `NPROC_PER_NODE=8` = **32 GPU** | 26 | 26×1×32 = **832** | `scripts/robodojo/train_video.sh:9`、`train_video_10k.yaml:6,23,162` |
| RoboDojo joint 0–100k | `NNODES=12` × 8 = **96 GPU** | 3 | 3×1×96 = **288** | `scripts/robodojo/train.sh:10`、`train_100k.yaml:4,27,271` |
| LIBERO Video-10k | `NPROC_PER_NODE=8` = **8 GPU** | 8 | 8×1×8 = **64** | `scripts/libero/train_video.sh:10`、`train_video.yaml:3,21,204` |
| LIBERO Action-30k | 8 GPU | 32 | 32×1×8 = **256** | `scripts/libero/train_action.sh:10`、`train_action.yaml:3,20,226` |
| RoboDojo 评测（参考） | **27 GPU / 19 policy server / 54 client** | — | 2100 episodes | `docs/evaluation.md:75`、`docs/robodojo.md:321-325` |

补充约束：
- `scripts/train.sh:31-37` 会**硬失败**：可见 GPU 数 ≠ `NPROC_PER_NODE` 时拒绝启动；
- `:38-41` 在 `REQUIRE_RDMA=1`（两个 RoboDojo 训练脚本的默认值）时要求
  `/dev/infiniband` 与 `ibv_devinfo`，即参考运行**用了 RDMA 网络**；
- 评测侧最低配置只需 2 张卡（1 张 policy + 1 张 Isaac Sim），
  但"参考测量"是 27 GPU / 19 server / 54 client（`docs/robodojo.md:321-325`）。

**显存相关**：仓库不公布显存需求，但训练器会在运行时测量峰值并校验
"峰值预留 ≤ (1 − `memory_headroom_fraction`) × 总显存"，默认余量 10%
（`trainer.py:756-772`、各配置 `preflight.memory_headroom_fraction: 0.1`）。
换言之，**能不能跑由 probe 阶段的 `probe.json` 决定**，而不是由文档承诺。

### 3.3 完整超参对照

| 超参 | 论文 | RoboDojo Video-10k | RoboDojo joint 100k | LIBERO Video | LIBERO Action |
|---|---|---|---|---|---|
| 全局 batch | Stage1 **1024** | 832 | 288 | 64 | 256 |
| 步数 | Stage1 **30,000** | 10,000 | 100,000 | 10,000 | 30,000 |
| 学习率 | Stage1 **1e-5** | 5e-6 | 视频 5e-6 / 动作 1e-4 | 5e-6 | 1e-4 |
| warmup | — | 200 | 500 | 200 | 500 |
| 调度 | — | cosine → 5e-8 | cosine → 0.01 倍，80k 后尾段到 0.001 倍 | cosine / floor 0.01 | cosine / floor 0.01 |
| AdamW betas | — | 0.9 / 0.999 | 0.9 / 0.95 | 0.9 / 0.999 | 0.9 / 0.95 |
| 精度 | — | bf16 | bf16 | bf16 | bf16 |
| ZeRO | — | 1 | 1 | 2 | 2 |
| 种子 | — | 100008 | 100008 | 42 | 42 |
| 帧数/分辨率 | 49 / 416×240 | 97→25 RGB / 240×416 | 65 时间戳→17 RGB / 240×416 | 65→17 RGB / 224×448 | 65→17 RGB / 224×448 |
| 保存间隔 | — | 每 2,000 | 每 5,000 | 每 2,000 | 每 5,000 |

**结论**：论文说的 Stage 1（bs 1024）与开源 recipe 的批次完全不是一回事；
开源 recipe 是**面向单/多机有限算力的后训练配方**，不能当作论文预训练的复现入口。
仓库也没有给出任何"训练需要多久"的估计。

---

## 4. 模型权重与数据可获得性

### 4.1 已开源的 checkpoint（`README.md:77-85`）

发布渠道：**HuggingFace `Haodong082399/VPP2`（公开）** 与 **ModelScope `haodong123/VPP2`（需账号）**
（`README.md:73-75`、`docs/robodojo.md:70-75`）。

| Checkpoint | 用途 | 蒸馏后？ | 备注 |
|---|---|---|---|
| `checkpoints_video/vpp2-video-stage1-49f.pth` | 零样本视频预测（事件级） | ❌ 蒸馏前 | 推理用 49 帧 + 30 步 |
| `checkpoints_video/vpp2-video-stage2-17f.pth` | 零样本视频预测（固定时程） | ❌ 蒸馏前 | 推理用 17 帧 + 30 步 |
| `checkpoints/initialization/robodojo_his10k.pt` | RoboDojo joint 训练的**起始**视频模型 | ❌ | 历史条件 Video-10k |
| `checkpoints/joint2b_s100000/video.pt` | RoboDojo 评测用视频骨干 | ❌ | 与下面 action.pt **必须配对** |
| `checkpoints/joint2b_s100000/action.pt` | RoboDojo 策略评测 | ❌ | 两者合计约 **36.85 GB**（`docs/robodojo.md:112`） |
| `checkpoints/libero/video_step010000.pt` | LIBERO 动作训练/评测 | ❌ | bf16，保留原始张量值 |
| `checkpoints/libero/action_step030000.pt` | LIBERO/OOD/PRO 评测 | ❌ | 含 proprio encoder，无优化器状态 |
| `checkpoints/Wan2.1-I2V-14B-480P/**` | 共享 VAE / CLIP / UMT5 / tokenizer | — | 训练与推理都需要 |

> 所有发布权重都是 **bf16 推理权重，不含优化器状态**（`docs/robodojo.md:113`、
> `docs/libero.md:78-80`）。因此**无法从发布的 checkpoint 继续训练**（续训需要完整 state）。

#### 4.1.1 每个文件到底多大（按参数量算，bf16 = 2 bytes/param）

发布权重统一用 bf16（`runtime.py:36-42` 默认 bf16；`policy.py:43-51`；
导出时 `weights.py:125` 保留原 dtype）。按第 2.1 节实测的参数量换算：

| 权重 | 参数量 | 体积 |
|---|---:|---:|
| 单个视频 DiT（Wan2.1-I2V-14B 微调后） | 16.39 B | **32.79 GB** |
| 动作专家 Action2B（RoboDojo） | 2.03 B | **4.06 GB** |
| 动作专家 ~1B（LIBERO，含 proprio） | 0.93 B | **1.86 GB** |

按"你要做什么"汇总下载量：

| 目标 | 需要下载 | 体积 |
|---|---|---|
| RoboDojo 策略评测 | `joint2b_s100000/{video,action}.pt` | **36.85 GB**（文档明写 36.85，计算值完全吻合） |
| LIBERO 训练 + 三种评测 | `libero/{video_step010000,action_step030000}.pt` | ≈ **34.6 GB**（仓库未标注） |
| 零样本视频预测（二选一） | `vpp2-video-stage1-49f.pth` 或 `stage2-17f.pth` | 各 ≈ **32.8 GB** |
| 从 Video-10k 起训 RoboDojo | `robodojo_his10k.pt` | ≈ **32.8 GB** |
| **五个视频级 ckpt 全收** | 上述全部 | ≈ **170 GB** |

#### 4.1.2 共享 Wan 编码器：只缺 4 个文件，不需要 Wan 的 14B DiT

`loader.py:160-165` 只要求 4 项：

| 文件 | 精度 | 体积（估算） |
|---|---|---:|
| `Wan2.1_VAE.pth` | fp32（`loader.py:178-181`） | ≈ 0.25 GB |
| `models_t5_umt5-xxl-enc-bf16.pth` | bf16 | ≈ 11.4 GB |
| `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` | fp16（只取 `visual.` + `log_scale`，`loader.py:200-213`） | ≈ 2–5 GB |
| `google/umt5-xxl/`（tokenizer） | — | ≈ 0.02 GB |
| **合计** | | **≈ 14–17 GB** |

⚠️ **体积数字来自上游 Wan2.1 资产的常识估计，仓库与论文都没有给出**，请以实际下载为准。

> **关键节省点**：**不需要**下载上游 Wan2.1 的 14B DiT 分片（约 32.8 GB）。
> 因为 VPP2 用自己的微调权重当 DiT——
> `policy.py:190 model_cfg.dit_checkpoint_path = str(video_path.resolve())`、
> `inference.py:281 dit_checkpoint_path=str(args.checkpoint)`，
> 上游 DiT 只会在既没给 `dit_checkpoint_path` 又没 glob 到
> `diffusion_pytorch_model-*.safetensors` 时才报错（`loader.py:144-158`）。
> 注意策略推理会强制加载文本编码器（`policy.py:192 model_cfg.load_text_encoder = True`），
> 所以 UMT5 那 11.4 GB 是省不掉的。

**显存粗算**（bf16 权重常驻）：视频 DiT 32.8 GB + UMT5 11.4 GB + CLIP ~2 GB + VAE 0.25 GB
≈ **46 GB 仅权重**，再加激活与 KV cache。所以 RoboDojo policy server 实际需要
**80 GB 级别的卡**（A100-80G / H100）。仓库不承诺任何显存要求——
按内部 `AGENTS.md` 的发布策略，只公布批次与步数（见 3.1）。

**全量下载总量**：≈ 170 GB（全部视频级 ckpt）+ 14–17 GB（编码器）≈ **185 GB 左右**。

### 4.2 **未**开源的权重

| 缺失项 | 说明 | 证据 |
|---|---|---|
| 论文主模型（Stage 1 事件级预训练权重） | 只放出了后续 Stage-1/2 视频权重，无法确认是否等价 | `README.md:101-103` |
| **蒸馏后的单步视频规划器** | 论文推理时延 0.12 s 依赖它；放出的都是蒸馏前版本 | `README.md:32` |
| LIBERO Stage-1 初始化骨干 `robot_video_step028000.pth` | 文档直接写 "remain pending release"，且"Raw Wan weights are not an equivalent initialization for reproducing the paper" | `docs/libero.md:113-115`、`docs/libero.md:184-186` |
| 真机 ALOHA 策略权重 | 未提及 | — |
| 大规模预训练数据与打标数据 | 未提供任何下载清单 | `README.md:101-103` |

### 4.3 数据集

| 数据集 | 仓库是否提供获取方式 |
|---|---|
| RoboDojo EE16 LeRobot v3.0（3500 episodes / 1,856,102 帧 / 25 Hz） | ✅ 固定 revision 的 HF 数据集 + `convert.sh`（`docs/training.md:33-50`） |
| LIBERO 四个 suite（1712 episodes / 277,713 帧） | 🟡 需自行用 MuJoCo 3.3.2 转换，**仓库不提供数据**，只提供 `prepare.sh` 校验（`docs/libero.md:136-148`） |
| LIBERO-OOD / LIBERO-PRO | 🟡 需自行 clone 指定 revision 的第三方仓库与资产（`docs/libero.md:307-347`） |
| 论文 Table 1/5 的大规模数据（OXE、DROID、Ego4D、OpenVid…） | ❌ 无任何获取指引 |

---

## 5. 论文表格可复现性矩阵

| 论文结果 | 本仓库能否复现 | 说明 |
|---|---|---|
| **Table 2** 视频指令遵循成功率（GPT 评估器） | ❌ | 没有 GPT 评估器代码，也没有 50 组评测对的构造脚本。只能主观跑 `infer_video.sh` 看效果 |
| **Figure 6** 人工成对胜率 | ❌ | 纯人工评测，无代码 |
| **Figure 7** 指令遵循定性对比 | 🟡 | 可自行用放出的视频权重生成，但**无基线权重**，无法对比 |
| **Figure 8** 真机 ALOHA 零样本（10 类） | ❌ | 无真机代码、无 ALOHA 数据、无 π0.5 / Fast-WAM 复现脚本、无策略权重 |
| **Table 3 / Table 6** LIBERO-ID | ✅ **可复现** | `configs/libero/eval_standard.yaml`（40 任务 × 50 init = 2000 episodes，`protocol: original2000`） |
| **Table 3** LIBERO-Pro（Position/Task） | ✅ **可复现** | `configs/libero/eval_pro.yaml`（80 cells × 10 = 800，`protocol: public800`） |
| **Table 3** LIBERO-OOD（Spatial/Object/Goal） | ✅ **可复现** | `configs/libero/eval_ood.yaml`（4500 episodes，`protocol: paper4500`） |
| **Table 4 / 7 / 8** RoboDojo 仿真的 Score 与 SR | ✅ **可复现** | `scripts/robodojo/{configure_eval,server,eval}.sh`；54 entries → 42 cells → 2100 episodes（`docs/evaluation.md:94-98`） |
| **Figure 9** VLM 子任务规划 | ❌ | 无闭环规划器、无子任务切分数据管线 |
| 训练侧：Stage 1 全过程 | ❌ | 未开源 |
| 训练侧：Stage 2 蒸馏 | ❌ | 未实现 |
| 训练侧：Stage 3（论文机制） | ❌ | LoRA 被禁用，改为全量微调/完全冻结 |
| 训练侧：RoboDojo joint 100k | ✅ **可完整重跑** | `configs/robodojo/train_100k.yaml` + 公开数据 + 发布初始化权重 |
| 训练侧：LIBERO Action-30k | 🟡 部分 | 可跑，但**缺少必须的 Stage-1 初始化骨干**（pending release），论文明确说原始 Wan 权重不等价 |
| 零样本视频预测 | ✅ 可跑（蒸馏前版本，30 步） | `scripts/infer_video.sh` |

> **仓库自测数字**：`docs/robodojo.md:374-375` 记录了发布 checkpoint 的官方口径结果——
> **680/2100** 成功、pooled SR **32.38%**、official SR **33.53%**、official Score **40.11**。
> 对比论文 Table 4 的 **39.26 / 32.26%**，仓库实测**略高**于论文数字（口径差异见下）。

### 5.1 评测协议细节（复现时要逐项对齐）

**LIBERO**（`src/vpp2/libero/protocols.py:4-59`）——三种协议都实现在同一套框架里：

| | standard（ID） | ood | pro |
|---|---|---|---|
| 协议名 | `original2000` | `paper4500` | `public800` |
| cells | 4 suite × 10 task = 40 | 3 seed ID × 3 `*_ood` suite × 10 = 90 | 4 suite × {swap, task} × 10 = 80 |
| 每 cell 试验数 | 50 | 50 | 10 |
| 总 episodes | 2000 | 4500 | 800 |
| policy seed | 42 | 7 | 42 |
| reset | 打包 init state 0–49 | 随机 | 打包 init state **1–10** |
| settle no-ops | 30 | 10 | 15 |
| 最大步数 | 400/400/400/700 | 330/420/450 | 220/280/300/520 |
| 试验种子 | sequential | `task_episode_hash`（SHA-256 v1） | 环境种子 1–10 |

统一推理契约（`libero/evaluate.py:94-125`）：`replan_steps=10`、`action_horizon=32`、
`num_inference_steps=10`、`sigma_shift=5.0`、`camera_preprocess_mode="training_exact"`、
`video_seed_offset=1`（视频高斯种子 = policy seed + 1）、`binarize_gripper=True`、
`text_cfg_scale=1.0`、tokenizer 128。

⚠️ **OOD 的种子哈希是仓库自己定的。** `libero/eval_helpers.py:195-213` 的注释明确说：
论文只要求"确定性哈希"但**没有公布具体哈希函数**，本仓库固定了一套可移植的
SHA-256 v1 方案（对 `(suite, task, episode, seed_id)` 求哈希后掩到 $2^{31}-1$）。
因此 OOD 数字**未必与论文逐 episode 一致**，只是同分布同预算。

⚠️ **PRO 使用"修复过的公开 runtime"**：`docs/libero.md:16-21` 说明用的是
HarnessVLA 公开 runtime 的 **direct-policy 800-episode 协议**（无 planner、无感知工具、
无任务记忆、无运动原语），并修复了 0.2.0 wheel 里 3 个畸形 Task BDDL 与不完整的 init 归档
（`prepare_pro.py:37-39`）。论文并未完整规定每个 reset/sampler 细节。

**RoboDojo**（`src/vpp2/robodojo.py:8-22`、`src/vpp2/benchmark/*`）：

- 推理契约 `INFERENCE = dict(shift=1.0, steps=10, horizon=32, replan=24, history_stride=25, seed=1)`
  中，`horizon` / `replan` / `history_stride` / `seed` 被**硬锁定**，只有 `shift` / `steps`
  可为了消融而覆盖（且必须配新的 run ID）；
- 54 entries → 42 official cells → **2100 episodes**；12 个 `*_random` entry 与对应 entry 配对，
  配对 entry 每项 25 episodes，其余 50（`scheduler_io.py:55-56`、`audit.py:148,227-228`）；
- 五个维度的 cell 数：Generalization 12、Precision 8、Long-Horizon 8、Memory 6、Open 8；
- 指标口径：official SR/Score = **先在每个维度内对 cell 取平均，再对五个维度等权平均**；
  pooled SR = 成功数 / 2100（两者不同，`docs/evaluation.md:100-104`）；
- 审计：`summary.json` / `audit.json` / `audit.md`，只有
  `complete=true` + 空 issue 列表 + `full_run.status == 0` 才算完成（`docs/libero.md:412-417` 同思路）。

---

## 6. 与论文的差异清单（按重要性排序）

| # | 差异 | 论文 | 代码 | 影响 |
|---|---|---|---|---|
| 1 | **推理不物化预测未来** | 单步生成视频 latent $F_\theta(\epsilon,0,c)$，动作条件于"预测的未来" | 默认路径只取视频专家在 $\sigma=1$（纯噪声）那一次前向的 K/V，**最终输出投影根本不计算**；一步视频 latent 仅作为可视化副产品、在动作去噪之后生成 | 视频专家在推理时更像"条件化特征提取器"而非显式规划器（见 2.6.1） |
| 2 | **蒸馏缺失** | 一致性蒸馏 → 单步，0.12 s | 无任何蒸馏实现；放出的权重是蒸馏前版本；零样本推理 30 步 | 论文最核心的时延主张无法复现或证伪（见 2.6.2） |
| 3 | **LoRA 缺失** | 冻结视频 DiT + 仅 LoRA 适配 | 明确抛异常拒绝 LoRA；RoboDojo 全量共训，LIBERO 完全冻结 | 论文第 1 节根因 2 的解法未被开源验证 |
| 4 | **Caption 简化** | 详细 caption（任务/末端/目标/视角四要素） | 训练与策略推理都只用 `"A video recorded from a robot's point of view executing the following instruction: {task}"` | 论文把 caption 当作泛化主因，开源 recipe 无法复现该增益 |
| 5 | **Stage 1 未开源** | 49 帧 / 30k 步 / bs 1024，事件级**均匀采样** | 无代码、无数据、无权重；`VideoEventDataset` 只有类名，采样是固定步长连续窗口 | 无法从零复现论文主模型 |
| 6 | **坐标对齐未开源** | 两级对齐 $A_d T_d B_d$，统一多数据集动作空间 | 仅单数据集内 z-score / min-max；`PoseRotationTransform` 是死代码 | 跨本体泛化的关键组件缺失 |
| 7 | **动作专家初始化不同** | "newly initialized" | 骨干由视频 DiT 插值 + α 缩放，仅 I/O 投影随机 | 与 LoRA 是两套不同的"保护泛化"思路 |
| 8 | **LIBERO 初始化缺失** | 从 prior robot-video step-28k 起 | 该文件标注 "remain pending release"；文档说原始 Wan 权重不等价 | LIBERO Stage 1 无法按论文复现 |
| 9 | **时间尺度不同** | Stage 2：机器人 8 s / 人手 2 s chunk | joint 未来 8 帧 × stride 4 = **1.28 s**；Video-10k 为 16 帧 = **2.56 s**；无 $H/f/\delta$ 换算，无 human-hand 训练数据 | 开源 recipe 是为 RoboDojo 重设计的，不是论文 Stage 2 直译 |
| 10 | **动作去噪步数** | 5 步去噪（0.1 s） | RoboDojo/LIBERO 推理都用 **10 步**（`robodojo.py:8`、`docs/libero.md:367`、`policy/VPP2/deploy.yml`） | 时延高于论文口径 |
| 11 | **无 `torch.compile`** | bf16 + torch.compile | 全库无 compile 调用 | 0.12 s 无法验证 |
| 12 | **视频构图异构** | T 型拼图 | RoboDojo 训练 `880×480→416×240`、推理 `440×240→416×240`；LIBERO 用水平 224×448 | 两条 recipe 不可互换 |
| 13 | **指标口径** | Table 4：Score 39.26 / SR 32.26% | 仓库实测 Score 40.11 / official SR 33.53% / pooled SR 32.38% | 非同一口径；仓库区分 pooled 与 official（`docs/robodojo.md:369-375`） |
| 14 | **论文自身小疑点** | Figure 1 右图出现 Table 4 中没有的 "G0.5(14.88)"，最高基线标 29.47 而 Table 4 是 28.97 | — | 疑点，非代码差异；建议向作者确认 |
| 15 | **Table 7/8 的 VPP2 行缺 Open 列** | PDF 文本层里 VPP2 行的 Open 值缺失 | 由 Avg 反推：Score ≈39.69、SR ≈32.33 | 排版/提取问题，需以原始表格为准 |

---

## 7. 今天到底能做什么

### ✅ 立刻可做

1. **跑零样本视频预测**：下载 Stage-1 (49f) 或 Stage-2 (17f) 视频权重 → `scripts/infer_video.sh`
   （注意是 30 步 + CFG 4，不是单步；可用 `scripts/enhance_prompt.sh` 做 Qwen 提示增强）。
2. **完整复现 LIBERO 三种评测**：下载 `video_step010000.pt` + `action_step030000.pt` +
   共享 Wan 资产 → 三个隔离仿真环境 → `scripts/libero/eval.sh` 依次跑
   standard / ood / pro，得到与论文 Table 3 同口径的 98.80 / 63.93 / 45.00。
3. **完整复现 RoboDojo 2100 episodes 评测**：下载配对权重（36.85 GB）→
   `configure_eval.sh` + `server.sh` + `eval.sh`，得到 Score/SR 与官方 audit。
4. **从公开数据重训 RoboDojo 策略**：`convert.sh` → `prepare.sh` → `text_cache.sh` →
   `train_video.sh`（可选，32 卡）→ `init_action.sh` → `train.sh`（96 卡）→ `export.sh`。
   这是仓库里**唯一一条端到端可复现的训练链**。

### ❌ 无法做

- 复现论文 Stage 1（大规模事件级预训练）与 Stage 2 的一致性蒸馏；
- 复现论文 Stage 3 的 LoRA 泛化保护机制；
- **验证论文 0.12 s / 0.1 s / 0.22 s 的时延主张**（无 `torch.compile`、无计时代码、动作 10 步而非 5 步）；
- **验证"动作条件于预测未来"这一机制细节**（默认推理路径不物化预测 latent，见 2.6.1）；
- 复现 Table 2 / Figure 6 的视频指令遵循定量评测（无 GPT 评估器与评测集）；
- 复现 Figure 8（真机 ALOHA 零样本）与 Figure 9（VLM 子任务规划）；
- 按论文方式复现 LIBERO Stage 1（初始化骨干 + 四 suite 训练数据均 pending release）；
- 从发布的推理权重继续训练（不含优化器状态）。

### 🔧 如果想最大化接近论文

1. 用 `enhance_prompt.py` 的 Qwen 增强结果**替换** `DEFAULT_PROMPT` 的简单模板，
   自己搭一个详细 caption 生成流程（对应论文附录 A.1 的四要素）；
   注意策略推理侧也要一并替换（`policy.py:400`），否则评测时仍然只见简单指令；
2. 自己实现一致性蒸馏；若不做，就接受 10 步去噪带来的时延；
3. 若目标是 RoboDojo 分数：直接用发布的 `joint2b_s100000` 配对权重，仓库自测
   40.11 / 33.53% 已是当前最好口径；
4. 若目标是泛化性研究：**必须**自己补齐 $\text{}^{e}T_d = A_d T_d B_d$ 的跨数据集对齐，
   否则无法验证论文第 2 节的核心设计；
5. 若要研究"视频预测到底有没有被用上"，从 `vpp2.py:1612` 的
   `capture_video_prediction` 开关入手做对照实验（开/关一步视频物化），
   这是本仓库里最容易做的、也最有价值的消融之一。

---

## 8. 附：关键文件索引

| 想看什么 | 去哪个文件 |
|---|---|
| 模型组装、MoT 前向、注意力掩码、视频缓存 | `src/vpp2/models/wan21_14b/vpp2.py`（1848 行；掩码 L878-906、缓存 L1002-1134、推理 L1466-1800） |
| MoT 混合注意力、专家拼接、视频 KV 预填充 | `src/vpp2/models/wan21_14b/mot.py`（957 行；联合注意力 L811-957） |
| 动作专家结构与骨干跳过清单 | `src/vpp2/models/wan21_14b/action_dit.py`（458 行；L37-43） |
| Flow matching / shift / 采样步 | `src/vpp2/models/wan21_14b/schedulers/scheduler_continuous.py` |
| 训练循环与可训练参数策略 | `src/vpp2/trainer.py:479-512`（`_apply_train_mode` / `_collect_trainable_params`） |
| RoboDojo 纯视频阶段 | `src/vpp2/video_training.py`、`configs/robodojo/train_video_10k.yaml` |
| 事件级数据集、历史帧采样、前缀替换 | `src/vpp2/datasets/lerobot/video_event_dataset.py`（历史 L728-842、前缀 L845-907） |
| 关键帧重采样（critical phase） | `src/vpp2/datasets/lerobot/critical_phase_*.py` |
| 归一化三种模式与截断 | `src/vpp2/datasets/lerobot/utils/normalizer.py` |
| 数据转换（EE16 校验 + T 型 ffmpeg） | `src/vpp2/convert_robodojo.py`、`src/vpp2/data.py:11-13` |
| Checkpoint 导出、配对校验、动作初始化 | `src/vpp2/weights.py`（L28-74、L108-161） |
| T 型拼图（推理侧） | `src/vpp2/utils/conditioning.py:52-80` |
| Qwen 提示增强 | `src/vpp2/enhance_prompt.py` |
| 策略推理入口（含 `capture_visual_plan`） | `src/vpp2/policy.py:392-420` |
| RoboDojo 推理契约（硬锁定） | `src/vpp2/robodojo.py:8-22` |
| 部署适配器 | `policy/VPP2/{model.py,deploy.yml}` |
| LIBERO 评测协议 | `src/vpp2/libero/{evaluate.py,protocols.py,pro_runtime.py}` |
| RoboDojo 评测与审计 | `src/vpp2/benchmark/{packed.py,audit.py}` |
| 发布范围策略（史料，已从公开版删除） | `git show e7e1a2a^:AGENTS.md` |
