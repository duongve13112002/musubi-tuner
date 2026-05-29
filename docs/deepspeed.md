# DeepSpeed ZeRO Training

Musubi Tuner supports [DeepSpeed ZeRO](https://www.deepspeed.ai/training/) stages 1, 2, and 3 for all training scripts **except WAN** (see [Known Limitations](#known-limitations)).

DeepSpeed is managed through [Accelerate](https://huggingface.co/docs/accelerate) — no changes to training script arguments are needed.

---

## Quick Start

### 1. Install DeepSpeed

```bash
pip install deepspeed
```

DeepSpeed requires Linux and CUDA. It does **not** install on Windows.

### 2. Configure Accelerate

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

### 3. Launch Training

```bash
accelerate launch --config_file accelerate_zero2.yaml \
    src/musubi_tuner/hv_train_network.py \
    --config your_training_config.toml
```

The script is the same as without DeepSpeed — Accelerate handles backend selection transparently.

---

## Choosing a ZeRO Stage

| Stage | What is sharded | Memory saving | Overhead | Recommendation |
|-------|----------------|---------------|----------|----------------|
| 1 | Optimizer states | Medium | Low | Safe default with 2+ GPUs |
| 2 | + Gradients | High | Low-medium | Best speed/memory trade-off |
| 3 | + Model parameters | Maximum | Higher | Only when GPU memory is critical |

**For LoRA fine-tuning**, Stage 2 is usually sufficient and gives excellent performance.
**For full fine-tuning** (`flux_2_train.py`, `qwen_image_train.py`, etc.), Stage 3 may be needed for very large models on limited VRAM.

---

## Config Templates

Ready-made JSON configs are in `docs/deepspeed_configs/`:

- **`zero2_bf16.json`** — ZeRO Stage 2, bfloat16. Safe choice for most setups.
- **`zero3_bf16.json`** — ZeRO Stage 3, bfloat16, no CPU offload. Adjust `reduce_bucket_size` and `stage3_prefetch_bucket_size` to tune GPU memory usage.

### CPU Offload (ZeRO-3 + Offload)

To offload optimizer states and parameters to CPU memory (useful for extremely large models on single-GPU setups), enable offloading in the config:

```json
"offload_optimizer": { "device": "cpu", "pin_memory": true },
"offload_param":     { "device": "cpu", "pin_memory": true }
```

CPU offload trades throughput for memory — expect 2–5× slower steps.

---

## Known Limitations

### WAN (`wan_train_network.py`) — Not Supported

WAN uses two transformer models simultaneously (the main DiT and a clip text encoder transformer). DeepSpeed via Accelerate wraps a single model, so WAN is fundamentally incompatible. Attempting to use DeepSpeed with WAN will raise a clear `ValueError` at startup.

Use standard DDP (`--multi_gpu`) for WAN multi-GPU training.

### `--blocks_to_swap` — Incompatible with ZeRO Stage 3

Block swapping (`--blocks_to_swap N`) and ZeRO Stage 3 both manage the same model parameters. Using both simultaneously raises a `ValueError`. This is intentional: the two memory strategies conflict.

Block swap is compatible with ZeRO Stage 1 and 2.

### Windows — DeepSpeed Not Installable

DeepSpeed's build system does not support Windows. Use WSL2 or a Linux machine for DeepSpeed training.

---

## How ZeRO Stage 3 Affects Checkpointing

Under ZeRO Stage 3, model parameters are sharded across GPUs. A single process's `model.state_dict()` only contains its local shard — not the full weights. Saving that shard produces a corrupt checkpoint.

Musubi Tuner automatically handles this: before every checkpoint save, it calls `accelerator.get_state_dict(model)` as a **collective operation** (all processes participate), which gathers all shards into a complete state dict on the main process. This happens transparently — you do not need to change your training command.

---

## Troubleshooting

### `"No module named 'deepspeed'"`

Install DeepSpeed: `pip install deepspeed`

### OOM even with ZeRO Stage 3

Try enabling CPU offload in the config (see above), or reduce `train_batch_size`.

### `"blocked_to_swap is incompatible with DeepSpeed ZeRO Stage 3"`

Remove `--blocks_to_swap` from your command, or switch to ZeRO Stage 1/2.

### Checkpoint appears corrupted / all-zeros

Ensure you are using `accelerate launch` and not `torchrun` directly. The `gather_state_dict_for_save` collective only works when Accelerate's DeepSpeed integration is active.
