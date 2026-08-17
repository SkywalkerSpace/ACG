# GR00T-N1 模型在 DexMimicGen 数据集上的微调与评估代码逻辑解析

本文档基于 ACG 项目中的两个核心脚本入口：
1. **训练入口**：`/Users/skywalker/code/experiment/ACG/libs/Isaac-GR00T-N1/scripts/gr00t_finetune_robocasa.py`
2. **测试/评估入口**：`/Users/skywalker/code/experiment/ACG/libs/Isaac-GR00T-N1/scripts/rollout_with_robomimic.py`

对 ACG 项目中基于 **DexMimicGen (DexMG)** 数据集的训练、模型架构、输入输出尺寸、微调方法及 ACG 引导测试逻辑进行深入全面剖析。

---

## 一、 代码整体架构与入口逻辑

项目采用**双大脑 (Dual Brain) 架构**：前级为多模态视觉-语言模型 (VLM Backbone, 基于 Eagle-2)，后级为连续动作生成头 (Action Head, 基于 Flow Matching + Cross-Attention DiT)。

```
[ 环境观察/数据集 HDF5 ]
        │
        ▼
[ DexmgDataset / MetaDexmgDataset ]
        │
        ▼
[ DexmgDataCollatorGR00T & GR00TTransform ]
        │
    ┌───┴────────────────────────────────────────┐
    │                                            │
    ▼                                            ▼
[ EagleBackbone (VLM) ]            [ State Encoder & Action Encoder ]
 图像 + 语言指令 -> 视觉语言 Embedding         状态 + 噪声动作 -> 状态/动作 Token
    │ (B, T_vlm, 1536)                           │ (B, 1+T_act, 1536)
    └─────────────────────┬──────────────────────┘
                          ▼
             [ Cross-Attention DiT Transformer ]
                          │
                          ▼
            [ Action Decoder (CategorySpecificMLP) ]
                          │
                          ▼
             [ 预测速度 (Velocity) / MSE Loss ] (训练)
                          或
           [ Euler 积分推导出的动作序列 (Action) ] (评估/测试)
```

---

## 二、 数据集与数据流水线 (DexMimicGen 核心)

### 1. 数据集定义 (`gr00t/data/dataset_robocasa.py`)

- **`DexmgDataset`**: 继承自 `robomimic.utils.dataset.SequenceDataset` 和 `LeRobotSetup`。
  - 负责从 HDF5 文件（如 `two_arm_threading.hdf5`）中读取轨迹序列，包括多摄像头图像、机械臂状态 (EEF pos/quat, joint qpos) 和动作。
  - 视频图像支持 256x256 分辨率的 HDF5 加载 (`videos_256x256/xxx_videos.hdf5`)。
- **`MetaDexmgDataset`**: 继承自 `MetaRobocasaDataset`，用于**多任务 / 多具身 (Multi-Embodiment)** 数据集的混合与动作归一化统计。
  - 在初始化时自动对不同具身形态分别计算并设置动作归一化参数 (`action_normalization_stats`)。

### 2. 跨具身形态映射 (Multi-Embodiment Mapping)

ACG 支持多种双手/多灵巧手机器人形态，在 DexMG 中主要包含以下具身 Tag (`embodiment_id`)：

| Tag / ID | Embodiment 名称 | 对应数据文件 (HDF5) | 动作空间组成 (Action Order) |
| :--- | :--- | :--- | :--- |
| **31** | `bimanual_panda_gripper` | `two_arm_threading.hdf5`<br>`two_arm_three_piece_assembly.hdf5`<br>`two_arm_transport.hdf5` | 右臂EEF(3)+姿态(3)+右夹爪(1)+<br>左臂EEF(3)+姿态(3)+左夹爪(1) **(共14维)** |
| **32** | `bimanual_panda_hand` | `two_arm_box_cleanup.hdf5`<br>`two_arm_drawer_cleanup.hdf5`<br>`two_arm_lift_tray.hdf5` | 右臂EEF(3)+姿态(3)+右灵巧手(6)+<br>左臂EEF(3)+姿态(3)+左灵巧手(6) **(共24维)** |
| **33** | `gr1_arms_only` | `two_arm_can_sort_random.hdf5` | 右臂(6)+左臂(6)+右手(6)+左手(6) **(共24维)** |
| **34** | `gr1_full_upper_body` (按arms_only处理) | `two_arm_coffee.hdf5`<br>`two_arm_pouring.hdf5` | 右臂(6)+左臂(6)+右手(6)+左手(6) **(共24维)** |

环境变量 `MAX_NUM_EMBODIMENTS` 在 DexMG 模式下设置为 **`35`**，以容纳不同的具身类别 ID。

### 3. 数据整理与预处理 (`gr00t/model/transforms.py` & `gr00t/experiment/data_config.py`)

- **`DexmgDataCollatorGR00T`**:
  1. 将 Robomimic/DexMG 格式的观察键名映射为 GR00T 标准键名（如 `agentview_image` -> `video.front_view`）。
  2. 提取连续帧 $T_o$（观察步长，默认 $T_o=1$），对视频进行双线性插值调整至 `(256, 256)` 并归一化到 `[0, 1]`。
  3. 提取未来 $T_p$ 步动作（预测步长，默认 $T_p=16$）。
  4. 支持根据 `null_ratio` 进行文本、状态或观察的 Classifier-Free Guidance (CFG) 空标数据随机丢弃（如将文本置为空串）。
- **`GR00TTransform`**:
  1. 调用 `EagleProcessor` 将多视角图像和文本提示词打包为 VLM 的 Prompt 格式。
  2. 将机械臂状态补齐 (Pad) 至固定最大维度 `max_state_dim = 64`，生成对应 `state_mask`。
  3. 将动作补齐 (Pad) 至固定最大维度 `max_action_dim = 32`，生成对应 `action_mask`。

---

## 三、 模型架构详解 (GR00T-N1 "Dual Brain")

`GR00T_N1` 由两大部分组成：

```
GR00T_N1
├── backbone: EagleBackbone (基于 Eagle-2 VLM)
└── action_head: FlowmatchingActionHead
    ├── state_encoder: CategorySpecificMLP
    ├── action_encoder: MultiEmbodimentActionEncoder
    ├── model: DiT (Cross-Attention Diffusion/Flow-Matching Transformer)
    └── action_decoder: CategorySpecificMLP
```

### 1. 视觉-语言主干 (`EagleBackbone`)
- **架构**：包含 Vision Tower（Vision Model）和预裁剪的 Transformer LLM 语言模型（裁剪为前 `select_layer = 12` 层）。
- **作用**：融合多视角图像 Token 与语言 Task Description，输出统一的多模态上下文特征 `backbone_features`。
- **降维/投影**：如配置了 `projector_dim`，过一层 Linear 映射为隐藏层维度（默认为 `1536`）。

### 2. 动作头 (`FlowmatchingActionHead`)
- **`CategorySpecificLinear` / `CategorySpecificMLP`**:
  - 为了同时支持多种不同关节/控制形态的机器人，模型的 State Encoder、Action Encoder 和 Action Decoder 均采用**具身类别特定层 (Category-Specific Layers)**。
  - 权重维度为 $(N_{categories}, D_{in}, D_{out})$，根据输入的 `embodiment_id`（如 31~34）动态选择对应机器人的专有权重矩阵。
- **`State Encoder`**:
  - 将 `max_state_dim=64` 维度的状态向量，通过对应具身 ID 的 MLP 映射至 `1536` 维。
- **`Action Encoder` (`MultiEmbodimentActionEncoder`)**:
  - 接收 $T_p=16$ 步加噪动作向量与离散化扩散时间步 $t$（0~1000）。
  - 将动作通过 `CategorySpecificLinear` 映射，结合 $t$ 的正弦位置编码 (`SinusoidalPositionalEncoding`)，经 Swish 激活后输出 `1536` 维动作特征。
- **`DiT` (`cross_attention_dit.py`)**:
  - 包含 12 层 `BasicTransformerBlock`。
  - **Self-Attention**: 作用于拼接后的 `[State_Tokens, Action_Tokens]` 序列（序列长度为 $1 + T_p = 17$）。
  - **Cross-Attention**: 将 VLM 的多模态特征 `backbone_features` 交叉注意力注入到动作/状态 Token 中。
  - **AdaLayerNorm**: 融合扩散时间步编码 $t_{emb}$ 进行 LayerNorm 调制。
- **`Action Decoder`**:
  - 将 DiT 输出的动作序列 Token，通过 `CategorySpecificMLP` 映射回 `action_dim`（默认 32，再截取实际具身维数）。

---

## 四、 模型不同部分输入与输出尺寸 (Tensor Shapes)

以 Batch Size $B=16$、预测步长 $T_p=16$、观察步长 $T_o=1$ 为例：

| 模块 / 步骤 | 输入 Tensor & 形状 | 输出 Tensor & 形状 | 说明 |
| :--- | :--- | :--- | :--- |
| **数据Collator** | `obs['agentview_image']`: `[B, T_o, H, W, 3]`<br>`actions`: `[B, T_p, D_act]` | `pixel_values`: `[B, V, 3, 256, 256]`<br>`input_ids`: `[B, N_tok]` | $V$ 为视角数 (通常2~3)<br>$N_{tok}$ 为文本+图像token数 |
| **EagleBackbone** | `pixel_values`: `[B, V, 3, 256, 256]`<br>`input_ids`: `[B, N_tok]` | `backbone_features`: `[B, N_seq, 1536]`<br>`backbone_attention_mask`: `[B, N_seq]` | $N_{seq}$ 为 VLM 编码后的上下文序列长度 |
| **State Encoder** | `state`: `[B, T_o, 64]`<br>`embodiment_id`: `[B]` | `state_features`: `[B, 1, 1536]` | 经 `CategorySpecificMLP` 编码 |
| **Action Encoder** | `noisy_trajectory`: `[B, T_p, 32]`<br>`t_discretized`: `[B]`<br>`embodiment_id`: `[B]` | `action_features`: `[B, T_p, 1536]` | 动作+时间步混合编码并加 Positional Embedding |
| **DiT (Diffusion)** | `sa_embs` = `cat(state, action)`: `[B, 17, 1536]`<br>`encoder_hidden_states`: `[B, N_seq, 1536]` | `model_output`: `[B, 17, 1024]` | 12层 Cross-Attention Transformer 块 |
| **Action Decoder** | `model_output`: `[B, 17, 1024]`<br>`embodiment_id`: `[B]` | `pred`: `[B, 17, 32]`<br>`pred_actions`: `[B, T_p, 32]` | 截取后 $T_p=16$ 步预测速度 $\hat{v}$ |
| **Loss 计算 (训练)** | `pred_actions`: `[B, T_p, 32]`<br>`velocity` ($x_1 - \text{noise}$): `[B, T_p, 32]` | `loss`: 标量 `()`, dtype `bf16/fp32` | MSE Loss 被 `action_mask` 遮罩 |
| **推理 (Inference)** | `actions` 初始噪声: `[B, T_p, 32]` | `action_pred`: `[B, T_p, 32]` | Euler 积分迭代 16 步更新最终生成的动作 |

---

## 五、 微调方法与训练参数

### 1. 训练入口控制 (`gr00t_finetune_robocasa.py`)

可以通过 Config 参数选择性冻结/训练模型的不同模块：
- `tune_llm` (`False`): 是否微调 VLM 的语言模型主干。
- `tune_visual` (`True`): 是否微调 VLM 的 Vision Tower。
- `tune_projector` (`True`): 是否微调 Action Head 的 State/Action Encoder 和 Action Decoder。
- `tune_diffusion_model` (`True`): 是否微调 Action Head 的 DiT 扩散模块。

### 2. PEFT / LoRA 参数高效微调支持 (`gr00t/utils/peft.py`)

若设置 `lora_rank > 0`（例如 `lora_rank = 32`）：
- 自动扫描模型中所有的 Attention 线性投影层（包含 `q_proj`, `v_proj`, `k_proj`, `to_q`, `to_v`, `to_k`）。
- 挂载 `peft.LoraConfig`（`lora_alpha=16`, `lora_dropout=0.1`），仅对 LoRA 旁路参数计算梯度。

### 3. 流匹配 (Flow Matching) 算法原理

模型动作生成基于 **Continuous Normalizing Flows (CNF) / Flow Matching**：
- **时间采提**: 采用 Beta 分布 $t_{sample} \sim \text{Beta}(\alpha=1.5, \beta=1.0)$，并按 $t = (s - t_{sample})/s$ 缩放。
- **轨迹插值**: 加噪轨迹为直线插值 $x_t = (1 - t) \cdot \epsilon + t \cdot x_1$，其中 $\epsilon \sim \mathcal{N}(0, I)$ 为标准高斯噪声，$x_1$ 为真实动作序列。
- **目标速度**: 真实向量场速度 $v = x_1 - \epsilon$。
- **损失函数**: 
  $$\mathcal{L}_{FM} = \frac{\sum | v_\theta(x_t, t, \text{obs}) - (x_1 - \epsilon) |^2 \cdot \text{mask}}{\sum \text{mask}}$$

### 4. 训练核心超参数一览

| 参数名 | 配置默认值 | 作用与说明 |
| :--- | :--- | :--- |
| `batch_size` | `16` | 每 GPU 训练批次大小 |
| `max_steps` | `10000` | 最大训练 Iteration 步数 |
| `learning_rate` | `1e-4` | 最大学习率 |
| `weight_decay` | `1e-5` | AdamW 权重衰减 |
| `warmup_ratio` | `0.05` | Cosine 调度器的预热比例 |
| `optim` | `"adamw_torch"` | 优化器 ($\beta_1=0.95, \beta_2=0.999$) |
| `bf16` / `tf32` | `True` | 开启 bfloat16 与 TensorFloat32 混合精度计算 |
| `save_steps` | `500` | Checkpoint 保存间隔 |
| `dataset_cls` | `"dexmg"` | 数据集类型，指定使用 `DexmgDataset` 和 `DexmgDataCollatorGR00T` |

---

## 六、 测试评估与 ACG (Action Classifier-Free Guidance) 逻辑

### 1. 评估流程 (`rollout_with_robomimic.py`)

1. 读取配置（如 `robocasa_mg100.json` 或 ACG 实验配置），通过 `EnvUtils.create_env_from_metadata` 构建 Isaac/Robomimic 仿真环境（支持 `tianshou.SubprocVectorEnv` 多进程并行）。
2. 使用 `algo_factory` 实例化算法策略对象（如 `gr00t_guidance_dexmg`）。
3. 使用 `RolloutPolicyWOLangEncoder` 包装策略模型。
4. 调用 `TrainUtils.rollout_with_stats` 循环执行评估。

### 2. 动作队列机制 (`action_queue`)

因为模型一次性预测未来 $T_p=16$ 步动作：
- 在环境评估时，当 `action_queue` 为空时，将当前时刻观察传入策略生成 $T_p$ 步动作。
- 取前 $T_a$ 步动作（Action Horizon，例如 $T_a=8$ 或 $16$）推入队列。
- 环境每一步 `step()` 时从 `action_queue` 弹出 1 步动作执行，大幅减少 VLM/DiT 推理频率。

### 3. ACG (Adaptive Cross-Attention Guidance / Action Classifier-Free Guidance) 原理与代码实现

ACG 论文提出的**非相干/扰动跨注意力引导 (Incoherent / Perturbed Cross-Attention Guidance)** 机制，源码位于 `robomimic/algo/guidance/acg.py`。

#### 核心逻辑 (`FlowmatchingActionHead_ACG`)：
在 Flow Matching 逆向采样过程（16步 Euler 积分）中：
1. **正常前向**: 计算无扰动的完整模型预测速度 $v_{\text{clean}} = v_\theta(x_t, t, \text{obs})$。
2. **扰动前向**: 若引导系数 $\gamma = \text{scale} \neq 1.0$，将 DiT 特定 Transformer 块（如 `skip_blocks = [7, 9, 11]`）中的 Cross-Attention Processor 替换为 `ACGAttnProcessor2_0`。
   - `ACGAttnProcessor2_0` 截断/破坏了来自 VLM 的 Key/Query scaled dot-product 计算，仅保留 Value 向量的线性投影，从而创造一个“信息缺失/不相干”的对比预测 $v_{\text{perturb}}$。
3. **引导组合**: 按照 ACG 引导公式外推最终方向：
   $$v_{\text{guided}} = v_{\text{clean}} + (\gamma - 1) \cdot (v_{\text{clean}} - v_{\text{perturb}})$$
4. **Euler 更新**:
   $$x_{t+\Delta t} = x_t + \Delta t \cdot v_{\text{guided}}$$

通过该引导技术，可以在不需要额外训练分类器的情况下，显著增强策略对语言指令和视觉上下文的依从度与动作生成的稳定性。

---

## 七、 总结

ACG 项目中的 GR00T-N1 结合了 **Eagle-2 VLM 强语义表征** 和 **Flow Matching DiT 强动作生成能力**：
- 在 **DexMimicGen** 数据集上，通过 `MetaDexmgDataset` 和 `CategorySpecificMLP/Linear` 实现了统一模型下的跨具身形态 (31~34 Tag) 联合微调。
- 在 **评估测试** 时，通过 `action_queue` 异步执行 $T_a$ 步动作，并结合 **ACG 跨注意力引导** 技术，大幅提升了复杂的双臂/灵巧手操纵任务的成功率。
