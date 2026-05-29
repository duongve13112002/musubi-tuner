> 📝 Click on the language section to expand / 言語をクリックして展開

# DeepSpeed ZeRO Training

## Overview / 概要

Musubi Tuner supports [DeepSpeed ZeRO](https://www.deepspeed.ai/training/) stages 1, 2, and 3 for all training scripts **except WAN** (see [Known Limitations](#known-limitations--既知の制限)).

DeepSpeed is managed through [Accelerate](https://huggingface.co/docs/accelerate) — no changes to training script arguments are needed.

<details>
<summary>日本語</summary>

Musubi TunerはWANを除くすべての学習スクリプトで、[DeepSpeed ZeRO](https://www.deepspeed.ai/training/) ステージ 1、2、3 をサポートしています（[既知の制限](#known-limitations--既知の制限)を参照）。

DeepSpeedは[Accelerate](https://huggingface.co/docs/accelerate)経由で管理されます。学習スクリプトの引数を変更する必要はありません。

</details>

## Quick Start / クイックスタート

### 1. Install DeepSpeed / DeepSpeedのインストール

```bash
pip install deepspeed
```

DeepSpeed requires Linux and CUDA. It does **not** install on Windows.

<details>
<summary>日本語</summary>

DeepSpeedはLinuxとCUDAが必要です。Windowsにはインストールできません。

</details>

### 2. Configure Accelerate / Accelerateの設定

Run `accelerate config` and select **DeepSpeed** as the distributed backend, then choose a ZeRO stage.

Alternatively, provide a config file:

```yaml
# accelerate_zero2.yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: docs/deepspeed_configs/zero2_bf16.json
  zero3_init_flag: false
num_processes: 2
```

For ZeRO Stage 3:

```yaml
# accelerate_zero3.yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: docs/deepspeed_configs/zero3_bf16.json
  zero3_init_flag: true
num_processes: 2
```

<details>
<summary>日本語</summary>

`accelerate config`を実行し、分散バックエンドとして**DeepSpeed**を選択してからZeROステージを選びます。

または、以下のように設定ファイルで指定することもできます。

ZeRO Stage 2の場合は`accelerate_zero2.yaml`、ZeRO Stage 3の場合は`accelerate_zero3.yaml`を作成し、上記のように記述します。

</details>

### 3. Launch Training / 学習の開始

```bash
accelerate launch --config_file accelerate_zero2.yaml \
    src/musubi_tuner/hv_train_network.py \
    --config your_training_config.toml
```

The script is the same as without DeepSpeed — Accelerate handles backend selection transparently.

<details>
<summary>日本語</summary>

スクリプトはDeepSpeedなしの場合と同じです。AccelerateがバックエンドをTransparentに処理します。

</details>

## Choosing a ZeRO Stage / ZeROステージの選択

| Stage | What is sharded / シャーディング対象 | Memory saving / 省メモリ | Overhead / オーバーヘッド | Recommendation / 推奨 |
|-------|-------------------------------------|--------------------------|---------------------------|------------------------|
| 1 | Optimizer states | Medium | Low | Safe default with 2+ GPUs |
| 2 | + Gradients | High | Low-medium | Best speed/memory trade-off |
| 3 | + Model parameters | Maximum | Higher | Only when GPU memory is critical |

**For LoRA fine-tuning**, Stage 2 is usually sufficient and gives excellent performance.
**For full fine-tuning** (`flux_2_train.py`, `qwen_image_train.py`, etc.), Stage 3 may be needed for very large models on limited VRAM.

<details>
<summary>日本語</summary>

**LoRAファインチューニング**の場合、ステージ2で十分で、優れたパフォーマンスが得られます。
**フルファインチューニング**（`flux_2_train.py`、`qwen_image_train.py`など）の場合、VRAMが限られた環境で非常に大きなモデルを扱うときはステージ3が必要になる場合があります。

</details>

## Config Templates / 設定テンプレート

Ready-made JSON configs are in `docs/deepspeed_configs/`:

- **`zero2_bf16.json`** — ZeRO Stage 2, bfloat16. Safe choice for most setups.
- **`zero3_bf16.json`** — ZeRO Stage 3, bfloat16, no CPU offload. Adjust `reduce_bucket_size` and `stage3_prefetch_bucket_size` to tune GPU memory usage.

### CPU Offload (ZeRO-3 + Offload) / CPUオフロード

To offload optimizer states and parameters to CPU memory (useful for extremely large models on single-GPU setups), enable offloading in the config:

```json
"offload_optimizer": { "device": "cpu", "pin_memory": true },
"offload_param":     { "device": "cpu", "pin_memory": true }
```

CPU offload trades throughput for memory — expect 2–5× slower steps.

<details>
<summary>日本語</summary>

`docs/deepspeed_configs/`にすぐ使えるJSONテンプレートが用意されています。

- **`zero2_bf16.json`** — ZeRO Stage 2、bfloat16。ほとんどの環境で安全に使用できます。
- **`zero3_bf16.json`** — ZeRO Stage 3、bfloat16、CPUオフロードなし。GPUメモリ使用量を調整するには`reduce_bucket_size`と`stage3_prefetch_bucket_size`を調整してください。

CPUオフロードはスループットとメモリのトレードオフです。ステップが2〜5倍遅くなることが予想されます。

</details>

## Known Limitations / 既知の制限

### WAN (`wan_train_network.py`) — Not Supported / 非対応

WAN uses two transformer models simultaneously (the main DiT and a clip text encoder transformer). DeepSpeed via Accelerate wraps a single model, so WAN is fundamentally incompatible. Attempting to use DeepSpeed with WAN will raise a clear `ValueError` at startup.

Use standard DDP (`--multi_gpu`) for WAN multi-GPU training.

<details>
<summary>日本語</summary>

WANはメインのDiTとclipテキストエンコーダーの2つのトランスフォーマーモデルを同時に使用します。AccelerateのDeepSpeedは単一モデルのラップしかサポートしていないため、WANとは根本的に非互換です。WANでDeepSpeedを使用しようとすると、起動時に明確な`ValueError`が発生します。

WANのマルチGPU学習には標準的なDDP（`--multi_gpu`）を使用してください。

</details>

### `--blocks_to_swap` — Incompatible with ZeRO Stage 3 / ZeRO Stage 3と非互換

Block swapping (`--blocks_to_swap N`) and ZeRO Stage 3 both manage the same model parameters. Using both simultaneously raises a `ValueError`. This is intentional: the two memory strategies conflict.

Block swap is compatible with ZeRO Stage 1 and 2.

<details>
<summary>日本語</summary>

ブロックスワッピング（`--blocks_to_swap N`）とZeRO Stage 3はどちらも同じモデルパラメータを管理します。両方を同時に使用すると`ValueError`が発生します。これは意図的な動作です。2つのメモリ戦略が競合するためです。

ブロックスワップはZeRO Stage 1および2と互換性があります。

</details>

### Windows — DeepSpeed Not Installable / Windowsにインストール不可

DeepSpeed's build system does not support Windows. Use WSL2 or a Linux machine for DeepSpeed training.

<details>
<summary>日本語</summary>

DeepSpeedのビルドシステムはWindowsをサポートしていません。DeepSpeed学習にはWSL2またはLinuxマシンを使用してください。

</details>

## How ZeRO Stage 3 Affects Checkpointing / ZeRO Stage 3がチェックポイントに与える影響

Under ZeRO Stage 3, model parameters are sharded across GPUs. A single process's `model.state_dict()` only contains its local shard — not the full weights. Saving that shard produces a corrupt checkpoint.

Musubi Tuner automatically handles this: before every checkpoint save, it calls `accelerator.get_state_dict(model)` as a **collective operation** (all processes participate), which gathers all shards into a complete state dict on the main process. This happens transparently — you do not need to change your training command.

<details>
<summary>日本語</summary>

ZeRO Stage 3では、モデルパラメータがGPU間でシャーディングされます。単一プロセスの`model.state_dict()`にはローカルシャードのみが含まれ、完全な重みは含まれません。そのシャードを保存すると、壊れたチェックポイントが生成されます。

Musubi Tunerはこれを自動的に処理します。チェックポイント保存の前に、**集団操作**として`accelerator.get_state_dict(model)`を呼び出し（すべてのプロセスが参加）、すべてのシャードをメインプロセス上の完全なstate dictに収集します。これは透過的に行われます。学習コマンドを変更する必要はありません。

</details>

## Troubleshooting / トラブルシューティング

### `"No module named 'deepspeed'"`

Install DeepSpeed: `pip install deepspeed`

### OOM even with ZeRO Stage 3 / ZeRO Stage 3でもOOM

Try enabling CPU offload in the config (see above), or reduce `train_batch_size`.

<details>
<summary>日本語</summary>

設定でCPUオフロードを有効にするか（上記参照）、`train_batch_size`を減らしてみてください。

</details>

### `"blocked_to_swap is incompatible with DeepSpeed ZeRO Stage 3"`

Remove `--blocks_to_swap` from your command, or switch to ZeRO Stage 1/2.

<details>
<summary>日本語</summary>

コマンドから`--blocks_to_swap`を削除するか、ZeRO Stage 1/2に切り替えてください。

</details>

### Checkpoint appears corrupted / all-zeros / チェックポイントが壊れている・全てゼロ

Ensure you are using `accelerate launch` and not `torchrun` directly. The `gather_state_dict_for_save` collective only works when Accelerate's DeepSpeed integration is active.

<details>
<summary>日本語</summary>

`torchrun`を直接使用せず、`accelerate launch`を使用していることを確認してください。`gather_state_dict_for_save`の集団操作はAccelerateのDeepSpeed統合が有効な場合にのみ機能します。

</details>
