> 📝 Click on the language section to expand / 言語をクリックして展開

# WAN 2.2 Full Fine-Tuning / WAN 2.2 フルファインチューニング

## Overview / 概要

`wan_train.py` provides full fine-tuning of WAN 2.2 DiT weights without LoRA. It supports training the low-noise model, the high-noise model, or both simultaneously via the `--train_models` flag.

WAN 2.2 uses a **dual-model denoising pipeline**: one `WanModel` handles high-noise timesteps (structure/composition, t ≥ 0.875) and another handles low-noise timesteps (detail/style, t < 0.875). Full fine-tuning lets you adapt the actual backbone weights of one or both models, rather than attaching adapter layers.

Pre-caching steps (latent cache, text-encoder output cache) are identical to WAN 2.2 LoRA training — reuse the same cache without re-running.

<details>
<summary>日本語</summary>

`wan_train.py`はLoRAなしでWAN 2.2のDiTの重みをフルファインチューニングします。`--train_models`フラグにより、低ノイズモデル、高ノイズモデル、または両方を同時に学習できます。

WAN 2.2は**デュアルモデルデノイジングパイプライン**を使用します。一方の`WanModel`が高ノイズタイムステップ（構造・構図、t ≥ 0.875）を、もう一方が低ノイズタイムステップ（ディテール・スタイル、t < 0.875）を処理します。フルファインチューニングにより、アダプターレイヤーを追加するのではなく、一方または両方のモデルの実際のバックボーンの重みを適応させることができます。

キャッシュ手順（潜在変数キャッシュ、テキストエンコーダー出力キャッシュ）はWAN 2.2 LoRA学習と同一です。再キャッシュ不要で同じキャッシュを再利用できます。

</details>

## Hardware Requirements / ハードウェア要件

WAN 2.2 A14B has approximately 11 billion parameters per model. Full fine-tuning memory requirements (per GPU):

| Configuration | Approx. VRAM needed |
|---|---|
| Both models, bf16, AdamW8bit, gradient checkpointing | ~80 GB+ (2–4 × A100/H100) |
| Low-noise only, bf16, AdamW8bit, gradient checkpointing | ~55 GB (single A100/H100) |
| Both models, bf16, AdamW8bit, `--blocks_to_swap 20` | ~48 GB (single A100) |
| Low-noise only, fp8 weights, AdamW8bit, `--blocks_to_swap 20` | ~24 GB (single A6000) |

> WAN is **not compatible with DeepSpeed**. Use standard DDP (`--multi_gpu`) for multi-GPU training.

<details>
<summary>日本語</summary>

WAN 2.2 A14Bはモデルあたり約110億パラメータです。フルファインチューニングのメモリ要件（GPU1枚あたり）：

| 構成 | 必要VRAMの目安 |
|---|---|
| 両モデル、bf16、AdamW8bit、勾配チェックポイント | ~80 GB以上（A100/H100 2〜4枚） |
| 低ノイズのみ、bf16、AdamW8bit、勾配チェックポイント | ~55 GB（A100/H100 1枚） |
| 両モデル、bf16、AdamW8bit、`--blocks_to_swap 20` | ~48 GB（A100 1枚） |
| 低ノイズのみ、fp8重み、AdamW8bit、`--blocks_to_swap 20` | ~24 GB（A6000 1枚） |

> WANはDeepSpeedと**非互換**です。マルチGPU学習には標準DDP（`--multi_gpu`）を使用してください。

</details>

## Quick Start / クイックスタート

### Step 1 — Cache latents (same as LoRA) / ステップ1 — 潜在変数キャッシュ（LoRAと同じ）

```bash
accelerate launch --num_cpu_threads_per_process 1 \
    src/musubi_tuner/wan_cache_latents.py \
    --task t2v-A14B \
    --vae path/to/wan_vae.safetensors \
    --dataset_config path/to/dataset.toml
```

### Step 2 — Cache text encoder outputs / ステップ2 — テキストエンコーダー出力キャッシュ

```bash
accelerate launch --num_cpu_threads_per_process 1 \
    src/musubi_tuner/wan_cache_text_encoder_outputs.py \
    --task t2v-A14B \
    --t5 path/to/t5_encoder \
    --dataset_config path/to/dataset.toml
```

### Step 3 — Train both models / ステップ3 — 両モデルの学習

```bash
accelerate launch --num_cpu_threads_per_process 1 \
    src/musubi_tuner/wan_train.py \
    --task t2v-A14B \
    --dit path/to/low_noise_model.safetensors \
    --dit_high_noise path/to/high_noise_model.safetensors \
    --vae path/to/wan_vae.safetensors \
    --t5 path/to/t5_encoder \
    --dataset_config path/to/dataset.toml \
    --sdpa --mixed_precision bf16 --full_bf16 \
    --train_models both \
    --optimizer_type adamw8bit --learning_rate 1e-5 \
    --gradient_checkpointing \
    --max_train_epochs 10 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output --output_name wan22-finetuned
```

### Step 3 (alternative) — Train low-noise model only / ステップ3（別案）— 低ノイズモデルのみ学習

```bash
accelerate launch --num_cpu_threads_per_process 1 \
    src/musubi_tuner/wan_train.py \
    --task t2v-A14B \
    --dit path/to/low_noise_model.safetensors \
    --vae path/to/wan_vae.safetensors \
    --t5 path/to/t5_encoder \
    --dataset_config path/to/dataset.toml \
    --sdpa --mixed_precision bf16 --full_bf16 \
    --train_models low \
    --optimizer_type adamw8bit --learning_rate 1e-5 \
    --gradient_checkpointing \
    --max_train_epochs 10 --save_every_n_epochs 1 \
    --output_dir path/to/output --output_name wan22-low-finetuned
```

<details>
<summary>日本語</summary>

**ステップ1 — 潜在変数キャッシュ**はLoRAと同じコマンドです。

**ステップ2 — テキストエンコーダー出力キャッシュ**も同様です。

**ステップ3 — 両モデルの学習**では`--train_models both`を指定し、低ノイズと高ノイズ両方のモデルパスを渡します。

**低ノイズのみ学習**の場合は`--train_models low`を指定し、`--dit`のみを渡します（`--dit_high_noise`は不要）。

</details>

## Choosing Which Models to Train / 学習するモデルの選択

### `--train_models {low,high,both}` / 学習モデルの指定

| Value | Models trained | Typical use case |
|---|---|---|
| `both` (default when both paths given) | Low-noise + High-noise | Full quality fine-tuning |
| `low` | Low-noise only (t < 0.875) | Style/character fine-tuning; most content appears in this range |
| `high` | High-noise only (t ≥ 0.875) | Composition/structure changes only |

**Smart default**: if only `--dit` is provided, `--train_models low` is assumed automatically. If only `--dit_high_noise` is provided, `--train_models high` is assumed. If both paths are provided, `--train_models both` is assumed.

When `--train_models low` is active and a high-noise timestep is sampled during training, that batch is skipped — the high-noise model is not updated. This is correct behavior; the data distribution for each model is respected.

<details>
<summary>日本語</summary>

**スマートデフォルト**：`--dit`のみ指定した場合は`low`が自動設定されます。`--dit_high_noise`のみの場合は`high`が自動設定されます。両方指定した場合は`both`が自動設定されます。

`--train_models low`が有効なときに高ノイズのタイムステップがサンプリングされた場合、そのバッチはスキップされます（高ノイズモデルは更新されません）。これは意図的な動作であり、各モデルのデータ分布を尊重しています。

</details>

## Learning Rate Configuration / 学習率の設定

You can specify different learning rates for the two models:

```bash
--learning_rate 1e-5          # fallback for both if per-model LR not set
--learning_rate_low 5e-5      # override LR for low-noise model only
--learning_rate_high 2e-5     # override LR for high-noise model only
```

If `--learning_rate_low` is not specified, it falls back to `--learning_rate`. Same for `--learning_rate_high`. This means you only need to set the ones you want to customise.

**Recommended**: start with the same LR for both models (`--learning_rate 1e-5`). Fine-tune per-model LRs after observing training dynamics.

<details>
<summary>日本語</summary>

2つのモデルに異なる学習率を指定できます。`--learning_rate_low`が未指定の場合は`--learning_rate`にフォールバックします。`--learning_rate_high`も同様です。

**推奨**：まず両モデルに同じ学習率（`--learning_rate 1e-5`）で開始し、学習の動向を観察してからモデルごとの学習率を調整してください。

</details>

## Checkpoint Output / チェックポイントの出力

### Dual model (`--train_models both`)

Two files are saved per checkpoint interval:

```
output_dir/
  wan22-finetuned_low-000001.safetensors   ← low-noise model weights
  wan22-finetuned_high-000001.safetensors  ← high-noise model weights
  wan22-finetuned_low.safetensors          ← final low-noise (after training ends)
  wan22-finetuned_high.safetensors         ← final high-noise
```

To use the checkpoints with inference or LoRA merging:

```bash
# Inference (wan_generate_video.py or similar)
--dit output_dir/wan22-finetuned_low.safetensors \
--dit_high_noise output_dir/wan22-finetuned_high.safetensors
```

### Single model (`--train_models low` or `high`)

A single file is saved (no `_low`/`_high` suffix):

```
output_dir/
  wan22-low-finetuned-000001.safetensors
  wan22-low-finetuned.safetensors
```

<details>
<summary>日本語</summary>

**デュアルモデル（`--train_models both`）**の場合、チェックポイント間隔ごとに2つのファイルが保存されます。サフィックス`_low`と`_high`によって区別されます。

推論やLoRAマージ時は`--dit`に低ノイズモデル、`--dit_high_noise`に高ノイズモデルを指定してください。

**シングルモデル（`--train_models low`または`high`）**の場合は、サフィックスなしの単一ファイルが保存されます。

</details>

## Sample Image Generation During Training / 学習中のサンプル画像生成

`--sample_prompts` is supported. For WAN 2.2 sampling, **both models are needed** — the full denoising pipeline routes timesteps between the two models.

If you are training only one model (`--train_models low`) but still want sample generation, provide the untrained model path too. The untrained model is loaded for inference only and its weights are not updated.

If either model is not available, sample generation is skipped with a warning.

<details>
<summary>日本語</summary>

`--sample_prompts`がサポートされています。WAN 2.2のサンプリングには**両モデルが必要**です。完全なデノイジングパイプラインがタイムステップに応じて2つのモデルにルーティングするためです。

`--train_models low`で低ノイズモデルのみを学習する場合でも、サンプル生成のために高ノイズモデルのパスも指定してください（学習は行われません）。

どちらかのモデルが利用できない場合、警告とともにサンプル生成はスキップされます。

</details>

## Memory Optimization / メモリ最適化

### Block swap / ブロックスワップ

`--blocks_to_swap N` offloads N transformer blocks to CPU during forward/backward passes. This applies to each model independently.

```bash
--blocks_to_swap 20
```

### Offload inactive model / 非アクティブモデルのオフロード

`--offload_inactive_dit` moves the model not currently being trained to CPU between forward passes. Reduces peak GPU memory at the cost of CPU↔GPU transfer time.

```bash
--offload_inactive_dit
```

### Memory-efficient saving / メモリ効率の良い保存

`--mem_eff_save` writes checkpoints one tensor at a time to reduce the memory spike at save time.

```bash
--mem_eff_save
```

<details>
<summary>日本語</summary>

**ブロックスワップ**：`--blocks_to_swap N`は各モデル独立してNブロックをCPUにオフロードします。

**非アクティブモデルのオフロード**：`--offload_inactive_dit`は現在学習中でないモデルをフォワードパス間でCPUに移動します。ピークVRAMを削減しますが、CPU↔GPU転送時間のコストがかかります。

**メモリ効率の良い保存**：`--mem_eff_save`は保存時のメモリスパイクを削減するため、テンソルを1つずつ書き込みます。

</details>

## Full CLI Reference / CLIリファレンス

All standard training arguments (`--learning_rate`, `--max_train_steps`, `--gradient_checkpointing`, etc.) are inherited from the common parser. WAN 2.2 specific arguments:

| Argument | Default | Description |
|---|---|---|
| `--train_models` | auto | `low`, `high`, or `both` — which model(s) to train |
| `--learning_rate_low` | `--learning_rate` | LR override for the low-noise model |
| `--learning_rate_high` | `--learning_rate` | LR override for the high-noise model |
| `--full_bf16` | `false` | Train DiT weights in bf16 (requires `--mixed_precision bf16`) |
| `--mem_eff_save` | `false` | Memory-efficient checkpoint saving |
| `--dit` | — | Path to low-noise model checkpoint |
| `--dit_high_noise` | — | Path to high-noise model checkpoint |
| `--timestep_boundary` | task default (875) | Timestep boundary (0–1000) splitting high/low noise models |
| `--offload_inactive_dit` | `false` | Move inactive model to CPU between steps |
| `--blocks_to_swap` | `0` | Number of transformer blocks to offload to CPU |
| `--task` | `t2v-14B` | Task config: `t2v-A14B`, `i2v-A14B` |

<details>
<summary>日本語</summary>

標準の学習引数（`--learning_rate`、`--max_train_steps`、`--gradient_checkpointing`等）は共通パーサーから継承されます。WAN 2.2固有の引数：

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--train_models` | 自動 | `low`、`high`、または`both` — 学習するモデルの選択 |
| `--learning_rate_low` | `--learning_rate` | 低ノイズモデルのLR上書き |
| `--learning_rate_high` | `--learning_rate` | 高ノイズモデルのLR上書き |
| `--full_bf16` | `false` | bf16でDiTの重みを学習（`--mixed_precision bf16`が必要） |
| `--mem_eff_save` | `false` | メモリ効率の良いチェックポイント保存 |
| `--dit` | — | 低ノイズモデルのチェックポイントパス |
| `--dit_high_noise` | — | 高ノイズモデルのチェックポイントパス |
| `--timestep_boundary` | タスクデフォルト（875） | 高/低ノイズモデルを分けるタイムステップ境界（0〜1000） |
| `--offload_inactive_dit` | `false` | 非アクティブモデルをステップ間でCPUに移動 |
| `--blocks_to_swap` | `0` | CPUにオフロードするTransformerブロック数 |
| `--task` | `t2v-14B` | タスク設定: `t2v-A14B`、`i2v-A14B` |

</details>

## Comparison with LoRA Fine-Tuning / LoRAとの比較

| | LoRA (`wan_train_network.py`) | Full Fine-Tuning (`wan_train.py`) |
|---|---|---|
| VRAM | Low (adapter only) | High (full model) |
| File size | Small (< 500 MB) | Large (20+ GB per model) |
| Training speed | Fast | Slower |
| Quality ceiling | Limited by LoRA rank | Full model expressiveness |
| Merge required | Yes (for deployment) | No (weights used directly) |
| DeepSpeed | Not supported | Not supported |

For most fine-tuning scenarios (style transfer, character consistency), **LoRA is recommended** due to lower VRAM requirements and faster iteration. Full fine-tuning is appropriate when the target domain shift is large or maximum quality is required.

<details>
<summary>日本語</summary>

| | LoRA (`wan_train_network.py`) | フルファインチューニング (`wan_train.py`) |
|---|---|---|
| VRAM | 低い（アダプターのみ） | 高い（フルモデル） |
| ファイルサイズ | 小さい（500MB未満） | 大きい（モデルあたり20GB以上） |
| 学習速度 | 速い | 遅い |
| 品質の上限 | LoRAランクに制限される | フルモデルの表現力 |
| マージの必要性 | あり（デプロイ時） | なし（重みを直接使用） |
| DeepSpeed | 非対応 | 非対応 |

ほとんどのファインチューニングシナリオ（スタイル転換、キャラクターの一貫性など）では、**LoRAが推奨**されます。VRAMの要件が低く、反復が速いためです。フルファインチューニングは、ターゲットドメインのシフトが大きい場合や最高品質が求められる場合に適しています。

</details>
