"""Full fine-tuning script for WAN 2.2 models (t2v-A14B, i2v-A14B).

Trains DiT parameters directly without LoRA. Supports training the low-noise
model only, the high-noise model only, or both simultaneously via the
--train_models flag.

Pre-caching scripts (wan_cache_latents.py, wan_cache_text_encoder_outputs.py)
and inference scripts are reused unchanged.

Usage example — both models (recommended for full quality):
    accelerate launch src/musubi_tuner/wan_train.py \\
        --task t2v-A14B \\
        --dit path/to/low_noise_model.safetensors \\
        --dit_high_noise path/to/high_noise_model.safetensors \\
        --vae path/to/wan_vae.safetensors \\
        --t5 path/to/t5_encoder \\
        --dataset_config path/to/dataset.toml \\
        --sdpa --mixed_precision bf16 --full_bf16 \\
        --train_models both \\
        --optimizer_type adamw8bit --learning_rate 1e-5 \\
        --gradient_checkpointing \\
        --max_train_epochs 10 --save_every_n_epochs 1 --seed 42 \\
        --output_dir path/to/output --output_name wan22-finetuned

Usage example — low-noise model only:
    accelerate launch src/musubi_tuner/wan_train.py \\
        --task t2v-A14B \\
        --dit path/to/low_noise_model.safetensors \\
        --vae path/to/wan_vae.safetensors \\
        --t5 path/to/t5_encoder \\
        --dataset_config path/to/dataset.toml \\
        --sdpa --mixed_precision bf16 --full_bf16 \\
        --train_models low \\
        --optimizer_type adamw8bit --learning_rate 1e-5 \\
        --gradient_checkpointing \\
        --max_train_epochs 10 --save_every_n_epochs 1 \\
        --output_dir path/to/output --output_name wan22-low-finetuned
"""

import argparse
import json
import math
from multiprocessing import Value
import os
import random
import time
from typing import Optional

import toml
import torch
from tqdm import tqdm
from accelerate import Accelerator
from safetensors.torch import save_file

from musubi_tuner.wan_train_network import WanNetworkTrainer, wan_setup_parser
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.hv_train_network import (
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_MINIMUM_KEYS,
    collator_class,
    clean_memory_on_device,
    prepare_accelerator,
    setup_parser_common,
    read_config_from_file,
    should_sample_images,
    set_seed,
)
from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3
from musubi_tuner.utils import huggingface_utils, model_utils, sai_model_spec, train_utils
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file
from musubi_tuner.training.deepspeed_utils import is_deepspeed_active, gather_state_dict_for_save

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _derive_train_flags(args: argparse.Namespace) -> tuple[bool, bool]:
    """Return (train_low, train_high) booleans from --train_models + available model paths."""
    tm = args.train_models

    has_low = bool(getattr(args, "dit", None))
    has_high = bool(getattr(args, "dit_high_noise", None))

    if tm is None:
        # smart default: train whatever is provided
        train_low = has_low
        train_high = has_high
    elif tm == "low":
        train_low = True
        train_high = False
    elif tm == "high":
        train_low = False
        train_high = True
    else:  # "both"
        train_low = True
        train_high = True

    if train_low and not has_low:
        raise ValueError(
            "--dit (low-noise model path) is required when train_models includes 'low'"
            " / train_modelsにlowが含まれる場合は--dit（低ノイズモデルパス）が必要です"
        )
    if train_high and not has_high:
        raise ValueError(
            "--dit_high_noise is required when train_models includes 'high'"
            " / train_modelsにhighが含まれる場合は--dit_high_noiseが必要です"
        )
    return train_low, train_high


def _make_ckpt_name(base_name: str, suffix: str, kind: str, step_or_epoch: Optional[int], is_step: bool, is_last: bool) -> str:
    """Build a checkpoint filename with optional _low/_high suffix.

    kind is '' for single-model or '_low'/'_high' for dual.
    """
    model_name = f"{base_name}{suffix}"
    if is_last:
        return train_utils.get_last_ckpt_name(model_name)
    if is_step:
        return train_utils.get_step_ckpt_name(model_name, step_or_epoch)
    return train_utils.get_epoch_ckpt_name(model_name, step_or_epoch)


class WanTrainer(WanNetworkTrainer):
    """Full fine-tuning trainer for WAN 2.2 dual-model architecture.

    Inherits model loading, inference, and VAE handling from WanNetworkTrainer.
    Overrides train() to train DiT parameters directly without LoRA, with
    support for training the low-noise model, high-noise model, or both.
    """

    def __init__(self):
        super().__init__()
        # used to pass the high-noise model reference into do_inference during sampling
        self._inference_transformer_high: Optional[torch.nn.Module] = None

    def load_transformer(
        self,
        accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device,
        dit_weight_dtype: Optional[torch.dtype],
    ) -> torch.nn.Module:
        """Override to load exactly one model without the LoRA dual-model internal load.

        WanNetworkTrainer.load_transformer embeds the LoRA state-dict swap mechanism:
        when high_low_training is True it always loads *both* models internally and
        stores the second as self.dit_inactive_state_dict.  For full fine-tuning we
        manage two separate model objects explicitly in wan_train.py, so that internal
        second load causes the high-noise model file to be read three times total on a
        dual-model launch (once wasted in the first call, twice in the second call).
        This override loads only the requested dit_path and skips the swap setup.
        """
        from musubi_tuner.wan.modules.model import load_wan_model

        model = load_wan_model(
            self.config,
            accelerator.device,
            dit_path,
            attn_mode,
            split_attn,
            loading_device,
            dit_weight_dtype,
            args.fp8_scaled,
            disable_numpy_memmap=args.disable_numpy_memmap,
        )
        if args.force_v2_1_time_embedding:
            model.set_time_embedding_v2_1(True)
        # Full fine-tune owns two model objects; no state-dict swap needed.
        self.dit_inactive_state_dict = None
        self.current_model_is_high_noise = False
        self.next_model_is_high_noise = False
        return model

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,  # low-noise model (or sole model)
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        """Dual-model denoising loop: routes each timestep to high or low model.

        For sampling to work correctly both models are needed. If the high-noise
        model is not available (self._inference_transformer_high is None), falls
        back to the single-model base implementation.
        """
        transformer_high = self._inference_transformer_high
        if transformer_high is None:
            # fall back: only low-noise model available, use parent implementation
            return super().do_inference(
                accelerator, args, sample_parameter, vae, dit_dtype, transformer,
                discrete_flow_shift, sample_steps, width, height, frame_count,
                generator, do_classifier_free_guidance, guidance_scale, cfg_scale,
                image_path=image_path, control_video_path=control_video_path,
            )

        from musubi_tuner.wan_generate_video import parse_one_frame_inference_args
        from musubi_tuner.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from musubi_tuner.hv_generate_video import resize_image_to_bucket
        from PIL import Image
        import numpy as np
        from musubi_tuner.dataset.image_video_dataset import load_video

        model_low = transformer
        model_high = transformer_high
        device = accelerator.device
        boundary = self.timestep_boundary if self.timestep_boundary is not None else self.config.boundary

        # ------ prepare geometry ------
        one_frame_mode = getattr(args, "one_frame", False)
        if one_frame_mode:
            target_index, control_indices, f_indices, one_frame_inference_index = parse_one_frame_inference_args(
                sample_parameter["one_frame"]
            )
            latent_video_length = len(f_indices)
        else:
            target_index, control_indices, f_indices, one_frame_inference_index = None, None, None, None
            latent_video_length = (frame_count - 1) // self.config["vae_stride"][0] + 1

        context = sample_parameter["t5_embeds"].to(device=device)
        if do_classifier_free_guidance:
            context_null = sample_parameter["negative_t5_embeds"].to(device=device)
        else:
            context_null = None

        num_channels_latents = 16
        vae_scale_factor = self.config["vae_stride"][1]
        lat_h = height // vae_scale_factor
        lat_w = width // vae_scale_factor

        # initialise latents
        shape_or_frame = (1, num_channels_latents, 1, lat_h, lat_w)
        latents_list = [torch.randn(shape_or_frame, generator=generator, device=device, dtype=torch.float32)
                        for _ in range(latent_video_length)]
        latents = torch.cat(latents_list, dim=2)
        image_latents = None

        # i2v / control setup (same as parent, condensed)
        if not one_frame_mode and (self.i2v_training or self.control_training):
            vae.to(device)
            vae.eval()
            if self.i2v_training:
                image = Image.open(image_path)
                image = resize_image_to_bucket(image, (width, height))
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(1).float()
                image = image / 127.5 - 1
                msk = torch.ones(1, frame_count, lat_h, lat_w, device=device)
                msk[:, 1:] = 0
                msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
                msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)
                with torch.amp.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
                    padding_frames = frame_count - 1
                    image = torch.concat([image, torch.zeros(3, padding_frames, height, width)], dim=1).to(device=device)
                    y = vae.encode([image])[0]
                y = y[:, :latent_video_length].unsqueeze(0)
                image_latents = torch.concat([msk, y], dim=1)
            if self.control_training:
                video = load_video(image_path or "", 0, frame_count, bucket_reso=(width, height))
                video = np.stack(video, axis=0)
                video = torch.from_numpy(video).permute(3, 0, 1, 2).float()
                video = (video / 127.5 - 1).to(device=device)
                with torch.amp.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
                    control_latents = vae.encode([video])[0][:, :latent_video_length].unsqueeze(0)
                if image_latents is not None:
                    image_latents = image_latents[:, 4:]
                    image_latents[:, :, 1:] = 0
                else:
                    image_latents = torch.zeros_like(control_latents)
                image_latents = torch.concat([control_latents, image_latents], dim=1)
            vae.to("cpu")
            clean_memory_on_device(device)

        scheduler = FlowUniPCMultistepScheduler(shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(sample_steps, device=device, shift=discrete_flow_shift)
        timesteps = scheduler.timesteps

        noise = torch.randn(16, latent_video_length, lat_h, lat_w, dtype=torch.float32,
                            generator=generator, device=device).to("cpu")

        max_seq_len = latent_video_length * lat_h * lat_w // (self.config.patch_size[1] * self.config.patch_size[2])
        arg_c = {"context": [context], "seq_len": max_seq_len}
        arg_null = {"context": [context_null], "seq_len": max_seq_len} if do_classifier_free_guidance else {}

        if self.i2v_training and not one_frame_mode:
            if not self.config.v2_2:
                clip_fea = sample_parameter["clip_embeds"].to(device=device, dtype=dit_dtype)
                arg_c["clip_fea"] = clip_fea
                if do_classifier_free_guidance:
                    arg_null["clip_fea"] = clip_fea
        if one_frame_mode:
            arg_c["f_indices"] = [f_indices]
            if do_classifier_free_guidance:
                arg_null["f_indices"] = [f_indices]
        if image_latents is not None:
            arg_c["y"] = image_latents
            if do_classifier_free_guidance:
                arg_null["y"] = image_latents

        prompt_idx = sample_parameter.get("enum", 0)
        latent = noise
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc=f"Sampling prompt {prompt_idx + 1}")):
                # route to the appropriate model based on current timestep
                t_norm = float(t.item()) / 1000.0
                current_model = model_high if t_norm >= boundary else model_low

                latent_model_input = [latent.to(device=device)]
                timestep = t.unsqueeze(0)

                with accelerator.autocast():
                    noise_pred_cond = current_model(latent_model_input, t=timestep, **arg_c)[0].to("cpu")
                    if do_classifier_free_guidance:
                        noise_pred_uncond = current_model(latent_model_input, t=timestep, **arg_null)[0].to("cpu")
                    else:
                        noise_pred_uncond = None

                if do_classifier_free_guidance:
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                temp_x0 = scheduler.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                                         return_dict=False, generator=generator)[0]
                latent = temp_x0.squeeze(0)

        vae.to(device)
        vae.eval()
        logger.info(f"Decoding video from latents: {latent.shape}")
        latent = latent.unsqueeze(0).to(device=device)
        if one_frame_mode:
            latent = latent[:, :, one_frame_inference_index : one_frame_inference_index + 1, :, :]
        with torch.amp.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
            video = vae.decode(latent)[0]
        video = video.unsqueeze(0).to(torch.float32).cpu()
        video = (video / 2 + 0.5).clamp(0, 1)
        vae.to("cpu")
        clean_memory_on_device(device)
        return video

    def _sample_images_dual(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        epoch,
        steps,
        vae,
        transformer_low: Optional[torch.nn.Module],
        transformer_high: Optional[torch.nn.Module],
        sample_parameters,
        dit_dtype: torch.dtype,
    ):
        """Sample images using both models. Skips if either model is missing."""
        if not should_sample_images(args, steps, epoch):
            return

        if transformer_low is None or transformer_high is None:
            logger.warning(
                "Skipping sample generation: both low-noise and high-noise models are required for WAN 2.2 sampling"
                " / サンプル生成をスキップ: WAN 2.2のサンプリングには低ノイズモデルと高ノイズモデルの両方が必要です"
            )
            return

        # store the high-noise model so do_inference can access it
        self._inference_transformer_high = accelerator.unwrap_model(transformer_high)
        try:
            self.sample_images(accelerator, args, epoch, steps, vae, transformer_low, sample_parameters, dit_dtype)
        finally:
            self._inference_transformer_high = None

    def train(self, args: argparse.Namespace):
        if torch.cuda.is_available():
            if args.cuda_allow_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 on CUDA / CUDAでTF32を有効化しました")
            if args.cuda_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmark / cuDNNベンチマークを有効化しました")

        # required arg check
        if args.dataset_config is None:
            raise ValueError("dataset_config is required / dataset_configが必要です")

        # determine which models to train
        train_low, train_high = _derive_train_flags(args)
        logger.info(f"train_models: low={train_low}, high={train_high}")

        # architecture-specific init (config, task, dtype detection)
        self.handle_model_specific_args(args)

        # override dtype for full fine-tune
        args.dit_dtype = "bfloat16" if args.full_bf16 else "float32"

        if args.show_timesteps:
            self.show_timesteps(args)
            return

        session_id = random.randint(0, 2**32)
        training_started_at = time.time()

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        if args.num_timestep_buckets is not None:
            logger.info(f"Using timestep bucketing. Number of buckets: {args.num_timestep_buckets}")
        self.num_timestep_buckets = args.num_timestep_buckets

        current_epoch = Value("i", 0)

        blueprint_generator = BlueprintGenerator(ConfigSanitizer())
        logger.info(f"Load dataset config from {args.dataset_config}")
        user_config = config_utils.load_user_config(args.dataset_config)
        blueprint = blueprint_generator.generate(user_config, args, architecture=self.architecture)
        train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
            blueprint.dataset_group, training=True, num_timestep_buckets=self.num_timestep_buckets,
            shared_epoch=current_epoch,
        )

        if train_dataset_group.num_train_items == 0:
            raise ValueError(
                "No training items found in the dataset. Please ensure the latent/Text Encoder cache has been created."
                " / データセットに学習データがありません。latent/Text Encoderキャッシュが作成済みか確認してください"
            )

        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = collator_class(current_epoch, ds_for_collator)

        logger.info("preparing accelerator")
        accelerator = prepare_accelerator(args)
        if args.mixed_precision is None:
            args.mixed_precision = accelerator.mixed_precision
            logger.info(f"mixed precision set to {args.mixed_precision}")
        is_main_process = accelerator.is_main_process

        # DeepSpeed is fundamentally incompatible with WAN (dual-model architecture).
        # This check is required here because WanTrainer.train() overrides the base
        # class train() entirely, so the deepspeed_supported guard in trainer_base.py
        # never runs.
        if is_deepspeed_active(accelerator):
            raise ValueError(
                "WAN does not support DeepSpeed. The dual-model architecture (low-noise + high-noise DiT) "
                "is fundamentally incompatible with DeepSpeed via Accelerate. "
                "Use standard DDP (no ACCELERATE_USE_DEEPSPEED) for multi-GPU training. "
                "/ WANはDeepSpeedをサポートしていません。デュアルモデルアーキテクチャはDeepSpeedと非互換です。"
                "マルチGPU学習には通常のDDP（デフォルト）を使用してください。"
            )

        if train_low and train_high and accelerator.num_processes > 1:
            logger.warning(
                "Multi-GPU (DDP) is not supported for dual-model training (--train_models both). "
                "Each GPU process samples timesteps independently, so different processes may route to "
                "different models per gradient-sync step, causing DDP all-reduce to hang. "
                "Use single-GPU training for --train_models both. "
                "/ デュアルモデル学習（--train_models both）ではマルチGPU（DDP）はサポートされていません。"
                "--train_models bothにはシングルGPUで実行してください。"
            )

        if train_low != train_high:
            logger.warning(
                "Single-model training mode: ~50%% of batches are skipped (timestep outside training range). "
                "Effective gradient updates ≈ max_train_steps / 2. "
                "To get N effective training epochs set --max_train_epochs = 2*N. "
                "/ シングルモデル学習モード: 約50%%のバッチがスキップされます。"
                "実効エポック数は指定値の約半分になります。--max_train_epochs = 2*N で N エポック相当になります。"
            )

        dit_dtype = model_utils.str_to_dtype(args.dit_dtype)
        vae_dtype = model_utils.str_to_dtype(args.vae_dtype)
        logger.info(f"DiT precision: {dit_dtype}, VAE precision: {vae_dtype}")

        # sample images setup
        sample_parameters = None
        vae = None
        if args.sample_prompts:
            sample_parameters = self.process_sample_prompts(args, accelerator, args.sample_prompts)
            vae = self.load_vae(args, vae_dtype=vae_dtype, vae_path=args.vae)
            vae.requires_grad_(False)
            vae.eval()

        # attention mode
        if args.sdpa:
            attn_mode = "torch"
        elif args.flash_attn:
            attn_mode = "flash"
        elif args.xformers:
            attn_mode = "xformers"
        elif args.flash3:
            attn_mode = "flash3"
        else:
            raise ValueError(
                "one of --sdpa / --flash_attn / --xformers / --flash3 must be specified"
                " / --sdpa / --flash_attn / --xformers / --flash3 のいずれかを指定してください"
            )

        blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0
        self.blocks_to_swap = blocks_to_swap

        # low-noise model (only loaded if needed)
        transformer_low = None
        if args.dit:
            loading_device_low = "cpu" if blocks_to_swap > 0 else accelerator.device
            logger.info(f"Loading low-noise DiT from {args.dit}")
            transformer_low = self.load_transformer(
                accelerator, args, args.dit, attn_mode, args.split_attn, loading_device_low, dit_dtype
            )
            if blocks_to_swap > 0:
                logger.info(f"Enabling block swap ({blocks_to_swap} blocks) for low-noise model")
                transformer_low.enable_block_swap(
                    blocks_to_swap, accelerator.device, supports_backward=True,
                    use_pinned_memory=args.use_pinned_memory_for_block_swap,
                )
                transformer_low.move_to_device_except_swap_blocks(accelerator.device)
            if train_low and args.gradient_checkpointing:
                transformer_low.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)

        # high-noise model (only loaded if needed for training or sampling)
        transformer_high = None
        if args.dit_high_noise:
            loading_device_high = "cpu" if blocks_to_swap > 0 else accelerator.device
            logger.info(f"Loading high-noise DiT from {args.dit_high_noise}")
            # reuse load_transformer: temporarily swap dit path arg
            original_dit = args.dit
            args.dit = args.dit_high_noise
            transformer_high = self.load_transformer(
                accelerator, args, args.dit_high_noise, attn_mode, args.split_attn, loading_device_high, dit_dtype
            )
            args.dit = original_dit
            if blocks_to_swap > 0:
                logger.info(f"Enabling block swap ({blocks_to_swap} blocks) for high-noise model")
                transformer_high.enable_block_swap(
                    blocks_to_swap, accelerator.device, supports_backward=True,
                    use_pinned_memory=args.use_pinned_memory_for_block_swap,
                )
                transformer_high.move_to_device_except_swap_blocks(accelerator.device)
            if train_high and args.gradient_checkpointing:
                transformer_high.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)

        accelerator.print("Preparing optimizers and data loaders / オプティマイザとデータローダを準備中")

        # learning rates: per-model with fallback to base learning_rate
        original_lr = args.learning_rate
        lr_low = getattr(args, "learning_rate_low", None) or original_lr
        lr_high = getattr(args, "learning_rate_high", None) or original_lr

        # build optimizers — temporarily override args.learning_rate for each call
        optimizer_low = optimizer_high = None
        optimizer_train_fn_low = optimizer_train_fn_high = lambda: None
        optimizer_eval_fn_low = optimizer_eval_fn_high = lambda: None
        optimizer_name_low = optimizer_name_high = ""
        optimizer_args_low = optimizer_args_high = ""

        if train_low and transformer_low is not None:
            args.learning_rate = lr_low
            optimizer_name_low, optimizer_args_low, optimizer_low, optimizer_train_fn_low, optimizer_eval_fn_low = (
                self.get_optimizer(args, list(transformer_low.parameters()))
            )
            logger.info(f"Low-noise optimizer: {optimizer_name_low}, lr={lr_low}")

        if train_high and transformer_high is not None:
            args.learning_rate = lr_high
            optimizer_name_high, optimizer_args_high, optimizer_high, optimizer_train_fn_high, optimizer_eval_fn_high = (
                self.get_optimizer(args, list(transformer_high.parameters()))
            )
            logger.info(f"High-noise optimizer: {optimizer_name_high}, lr={lr_high}")

        args.learning_rate = original_lr  # restore

        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs: {args.max_train_steps}"
            )

        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # LR schedulers
        lr_scheduler_low = lr_scheduler_high = None
        if train_low and optimizer_low is not None:
            lr_scheduler_low = self.get_lr_scheduler(args, optimizer_low, accelerator.num_processes)
        if train_high and optimizer_high is not None:
            lr_scheduler_high = self.get_lr_scheduler(args, optimizer_high, accelerator.num_processes)

        args.full_fp16 = False
        if args.full_bf16:
            assert args.mixed_precision == "bf16", (
                "full_bf16 requires mixed_precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください"
            )
            accelerator.print("Enabled full bf16 training / full bf16学習を有効化しました")

        # prepare with accelerator
        if train_low and transformer_low is not None:
            if blocks_to_swap > 0:
                transformer_low = accelerator.prepare(transformer_low, device_placement=[False])
                accelerator.unwrap_model(transformer_low).move_to_device_except_swap_blocks(accelerator.device)
                accelerator.unwrap_model(transformer_low).prepare_block_swap_before_forward()
            else:
                transformer_low = accelerator.prepare(transformer_low)
            if args.compile:
                transformer_low = self.compile_transformer(args, transformer_low)
                transformer_low.__dict__["_orig_mod"] = transformer_low
            optimizer_low, lr_scheduler_low = accelerator.prepare(optimizer_low, lr_scheduler_low)

        if train_high and transformer_high is not None:
            if blocks_to_swap > 0:
                transformer_high = accelerator.prepare(transformer_high, device_placement=[False])
                accelerator.unwrap_model(transformer_high).move_to_device_except_swap_blocks(accelerator.device)
                accelerator.unwrap_model(transformer_high).prepare_block_swap_before_forward()
            else:
                transformer_high = accelerator.prepare(transformer_high)
            if args.compile:
                transformer_high = self.compile_transformer(args, transformer_high)
                transformer_high.__dict__["_orig_mod"] = transformer_high
            optimizer_high, lr_scheduler_high = accelerator.prepare(optimizer_high, lr_scheduler_high)

        train_dataloader = accelerator.prepare(train_dataloader)

        self.resume_from_local_or_hf_if_specified(accelerator, args)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num train items / 学習アイテム数: {train_dataset_group.num_train_items}")
        accelerator.print(f"  num batches per epoch: {len(train_dataloader)}")
        accelerator.print(f"  num epochs: {num_train_epochs}")
        accelerator.print(f"  gradient accumulation steps: {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps: {args.max_train_steps}")
        accelerator.print(f"  train_models: low={train_low}, high={train_high}")
        accelerator.print(f"  lr_low={lr_low}, lr_high={lr_high}")

        # checkpoint suffix: add _low/_high only when training both
        suffix_low = "_low" if (train_low and train_high) else ""
        suffix_high = "_high" if (train_low and train_high) else ""

        # metadata
        metadata = {
            "ss_session_id": session_id,
            "ss_training_started_at": training_started_at,
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_learning_rate_low": lr_low,
            "ss_learning_rate_high": lr_high,
            "ss_train_models": args.train_models or "auto",
            "ss_num_train_items": train_dataset_group.num_train_items,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_checkpointing_cpu_offload": args.gradient_checkpointing_cpu_offload,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            SS_METADATA_KEY_BASE_MODEL_VERSION: self.architecture_full_name,
            "ss_mixed_precision": args.mixed_precision,
            "ss_seed": args.seed,
            "ss_training_comment": args.training_comment,
            "ss_optimizer": optimizer_name_low + (f"({optimizer_args_low})" if optimizer_args_low else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_full_bf16": bool(args.full_bf16),
            "ss_weighting_scheme": args.weighting_scheme,
            "ss_logit_mean": args.logit_mean,
            "ss_logit_std": args.logit_std,
            "ss_mode_scale": args.mode_scale,
            "ss_timestep_sampling": args.timestep_sampling,
            "ss_discrete_flow_shift": args.discrete_flow_shift,
            "ss_task": args.task,
        }

        datasets_metadata = [ds.get_metadata() for ds in train_dataset_group.datasets]
        metadata["ss_datasets"] = json.dumps(datasets_metadata)

        if args.dit:
            sd_model_name = os.path.basename(args.dit) if os.path.exists(args.dit) else args.dit
            metadata["ss_sd_model_name"] = sd_model_name
        if args.dit_high_noise:
            hn_name = os.path.basename(args.dit_high_noise) if os.path.exists(args.dit_high_noise) else args.dit_high_noise
            metadata["ss_dit_high_noise_name"] = hn_name
        if args.vae:
            vae_name = os.path.basename(args.vae) if os.path.exists(args.vae) else args.vae
            metadata["ss_vae_name"] = vae_name

        metadata = {k: str(v) for k, v in metadata.items()}
        minimum_metadata = {k: metadata[k] for k in SS_METADATA_MINIMUM_KEYS if k in metadata}

        if is_main_process:
            init_kwargs = {}
            if args.wandb_run_name:
                init_kwargs["wandb"] = {"name": args.wandb_run_name}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            accelerator.init_trackers(
                "fine-tuning" if args.log_tracker_name is None else args.log_tracker_name,
                config=train_utils.get_sanitized_config_or_none(args),
                init_kwargs=init_kwargs,
            )

        # helper: save one model checkpoint
        def save_model_ckpt(
            ckpt_name: str,
            unwrapped_model,
            steps: int,
            epoch_no: int,
            precomputed_state=None,
            use_mem_eff: bool = False,
        ):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)
            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata

            title = args.metadata_title if args.metadata_title is not None else args.output_name
            if args.min_timestep is not None or args.max_timestep is not None:
                md_ts = (args.min_timestep or 0, args.max_timestep or 1000)
            else:
                md_ts = None
            sai_metadata = sai_model_spec.build_metadata(
                None, self.architecture, time.time(), title,
                args.metadata_reso, args.metadata_author, args.metadata_description,
                args.metadata_license, args.metadata_tags,
                timesteps=md_ts, is_lora=False, custom_arch=args.metadata_arch,
            )
            metadata_to_save.update(sai_metadata)

            if precomputed_state is not None:
                state_dict = precomputed_state
                if any("_orig_mod." in k for k in state_dict.keys()):
                    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            else:
                # torch.compile stores compiled state under _orig_mod. keys; strip the prefix
                # so the saved checkpoint is compatible with non-compiled loading.
                state_dict = unwrapped_model.state_dict()
                if any("_orig_mod." in k for k in state_dict.keys()):
                    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

            if use_mem_eff:
                mem_eff_save_file(state_dict, ckpt_file, metadata_to_save)
            else:
                save_file(state_dict, ckpt_file, metadata_to_save)

            if args.huggingface_repo_id is not None:
                huggingface_utils.upload(args, ckpt_file, "/" + ckpt_name)

        def remove_model_ckpt(old_ckpt_name: str):
            old_path = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_path):
                accelerator.print(f"removing old checkpoint: {old_path}")
                os.remove(old_path)

        def save_both(ckpt_name_low: str, ckpt_name_high: str, step: int, epoch: int,
                      state_low=None, state_high=None):
            """Gather + save both models; gathering is a collective operation (all processes)."""
            if train_low and transformer_low is not None:
                state = state_low if state_low is not None else gather_state_dict_for_save(accelerator, transformer_low)
                accelerator.wait_for_everyone()
                if is_main_process:
                    save_model_ckpt(ckpt_name_low, accelerator.unwrap_model(transformer_low), step, epoch,
                                    precomputed_state=state, use_mem_eff=args.mem_eff_save)
            if train_high and transformer_high is not None:
                state = state_high if state_high is not None else gather_state_dict_for_save(accelerator, transformer_high)
                accelerator.wait_for_everyone()
                if is_main_process:
                    save_model_ckpt(ckpt_name_high, accelerator.unwrap_model(transformer_high), step, epoch,
                                    precomputed_state=state, use_mem_eff=args.mem_eff_save)

        # initial sample
        if should_sample_images(args, 0, epoch=0):
            optimizer_eval_fn_low()
            optimizer_eval_fn_high()
            self._sample_images_dual(accelerator, args, 0, 0, vae,
                                     transformer_low, transformer_high, sample_parameters, dit_dtype)
            optimizer_train_fn_low()
            optimizer_train_fn_high()
        if len(accelerator.trackers) > 0:
            accelerator.log({}, step=0)

        # log initial dtype/device
        if transformer_low is not None:
            uw = accelerator.unwrap_model(transformer_low)
            p = next(iter(uw.parameters()), None)
            logger.info(f"Low DiT dtype: {p.dtype if p is not None else None}, device: {p.device if p is not None else accelerator.device}")
        if transformer_high is not None:
            uw = accelerator.unwrap_model(transformer_high)
            p = next(iter(uw.parameters()), None)
            logger.info(f"High DiT dtype: {p.dtype if p is not None else None}, device: {p.device if p is not None else accelerator.device}")

        clean_memory_on_device(accelerator.device)
        optimizer_train_fn_low()
        optimizer_train_fn_high()

        progress_bar = tqdm(range(args.max_train_steps), smoothing=0,
                            disable=not accelerator.is_local_main_process, desc="steps")
        global_step = 0
        noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")
        loss_recorder = train_utils.LossRecorder()
        del train_dataset_group

        timestep_boundary = self.timestep_boundary if self.timestep_boundary is not None else self.config.boundary

        for epoch in range(num_train_epochs):
            accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}")
            current_epoch.value = epoch + 1
            metadata["ss_epoch"] = str(epoch + 1)

            for step, batch in enumerate(train_dataloader):
                latents = batch["latents"]

                # determine active model from timestep before accumulation scope
                noise = torch.randn_like(latents)
                noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(
                    args, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype
                )

                # route: is this batch for the high-noise or low-noise model?
                t_norm = float(timesteps[0].item()) / 1000.0
                is_high_noise = t_norm >= timestep_boundary

                if is_high_noise and not train_high:
                    continue  # skip batches outside training range
                if not is_high_noise and not train_low:
                    continue

                # select active model + optimizer
                if is_high_noise:
                    active_model = transformer_high
                    active_opt = optimizer_high
                    active_sched = lr_scheduler_high
                    opt_train_fn = optimizer_train_fn_high
                    opt_eval_fn = optimizer_eval_fn_high
                else:
                    active_model = transformer_low
                    active_opt = optimizer_low
                    active_sched = lr_scheduler_low
                    opt_train_fn = optimizer_train_fn_low
                    opt_eval_fn = optimizer_eval_fn_low

                # offload inactive model to CPU if requested
                if args.offload_inactive_dit:
                    inactive = transformer_high if not is_high_noise else transformer_low
                    if inactive is not None:
                        accelerator.unwrap_model(inactive).to("cpu")
                    accelerator.unwrap_model(active_model).to(accelerator.device)

                with accelerator.accumulate(active_model):
                    latents = self.scale_shift_latents(latents)

                    weighting = compute_loss_weighting_for_sd3(
                        args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, dit_dtype
                    )

                    output = self._call_dit(
                        args, accelerator, active_model, latents, batch, noise,
                        noisy_model_input, timesteps, dit_dtype
                    )
                    loss = torch.nn.functional.mse_loss(
                        output.pred.to(dit_dtype), output.target.to(dit_dtype), reduction="none"
                    )
                    if weighting is not None:
                        loss = loss * weighting
                    loss = loss.mean()

                    accelerator.backward(loss)

                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        accelerator.clip_grad_norm_(active_model.parameters(), args.max_grad_norm)

                    active_opt.step()
                    active_sched.step()
                    active_opt.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    if global_step == 0:
                        progress_bar.reset()
                    progress_bar.update(1)
                    global_step += 1

                    should_sampling = should_sample_images(args, global_step, epoch=None)
                    should_saving = args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0

                    if should_sampling or should_saving:
                        # Always switch BOTH optimizers to eval — sampling and saving use both
                        # models regardless of which model was active for this step.
                        optimizer_eval_fn_low()
                        optimizer_eval_fn_high()
                        if should_sampling:
                            self._sample_images_dual(accelerator, args, None, global_step, vae,
                                                     transformer_low, transformer_high, sample_parameters, dit_dtype)
                        if should_saving:
                            ckpt_low = _make_ckpt_name(args.output_name, suffix_low, "low", global_step, True, False)
                            ckpt_high = _make_ckpt_name(args.output_name, suffix_high, "high", global_step, True, False)
                            save_both(ckpt_low, ckpt_high, global_step, epoch)
                            if is_main_process:
                                remove_no = train_utils.get_remove_step_no(args, global_step)
                                if remove_no is not None:
                                    if train_low:
                                        remove_model_ckpt(_make_ckpt_name(args.output_name, suffix_low, "low", remove_no, True, False))
                                    if train_high:
                                        remove_model_ckpt(_make_ckpt_name(args.output_name, suffix_high, "high", remove_no, True, False))
                        optimizer_train_fn_low()
                        optimizer_train_fn_high()

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}
                progress_bar.set_postfix(**logs)

                if len(accelerator.trackers) > 0:
                    model_tag = "high" if is_high_noise else "low"
                    accelerator.log({"loss": current_loss, "avr_loss": avr_loss, "active_model": model_tag},
                                    step=global_step)

                if global_step >= args.max_train_steps:
                    break

            if len(accelerator.trackers) > 0:
                accelerator.log({"loss/epoch": loss_recorder.moving_average}, step=epoch + 1)

            accelerator.wait_for_everyone()
            optimizer_eval_fn_low()
            optimizer_eval_fn_high()

            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                # gather is collective — must happen on all processes even if not saving
                state_low = gather_state_dict_for_save(
                    accelerator, transformer_low, should_save=saving
                ) if (train_low and transformer_low is not None) else None
                state_high = gather_state_dict_for_save(
                    accelerator, transformer_high, should_save=saving
                ) if (train_high and transformer_high is not None) else None

                if is_main_process and saving:
                    ckpt_low = _make_ckpt_name(args.output_name, suffix_low, "low", epoch + 1, False, False)
                    ckpt_high = _make_ckpt_name(args.output_name, suffix_high, "high", epoch + 1, False, False)
                    if train_low and transformer_low is not None:
                        save_model_ckpt(ckpt_low, accelerator.unwrap_model(transformer_low),
                                        global_step, epoch + 1, precomputed_state=state_low,
                                        use_mem_eff=args.mem_eff_save)
                        remove_no = train_utils.get_remove_epoch_no(args, epoch + 1)
                        if remove_no is not None:
                            remove_model_ckpt(_make_ckpt_name(args.output_name, suffix_low, "low", remove_no, False, False))
                    if train_high and transformer_high is not None:
                        save_model_ckpt(ckpt_high, accelerator.unwrap_model(transformer_high),
                                        global_step, epoch + 1, precomputed_state=state_high,
                                        use_mem_eff=args.mem_eff_save)
                        remove_no = train_utils.get_remove_epoch_no(args, epoch + 1)
                        if remove_no is not None:
                            remove_model_ckpt(_make_ckpt_name(args.output_name, suffix_high, "high", remove_no, False, False))

            self._sample_images_dual(accelerator, args, epoch + 1, global_step, vae,
                                     transformer_low, transformer_high, sample_parameters, dit_dtype)
            optimizer_train_fn_low()
            optimizer_train_fn_high()

        metadata["ss_training_finished_at"] = str(time.time())

        # ZeRO3 gather — must be collective (all processes), before end_training
        final_state_low = gather_state_dict_for_save(
            accelerator, transformer_low
        ) if (train_low and transformer_low is not None) else None
        final_state_high = gather_state_dict_for_save(
            accelerator, transformer_high
        ) if (train_high and transformer_high is not None) else None

        if is_main_process:
            if transformer_low is not None:
                transformer_low = accelerator.unwrap_model(transformer_low)
            if transformer_high is not None:
                transformer_high = accelerator.unwrap_model(transformer_high)

        accelerator.end_training()

        optimizer_eval_fn_low()
        optimizer_eval_fn_high()

        if is_main_process:
            ckpt_low = _make_ckpt_name(args.output_name, suffix_low, "low", None, False, True)
            ckpt_high = _make_ckpt_name(args.output_name, suffix_high, "high", None, False, True)
            if train_low and transformer_low is not None:
                save_model_ckpt(ckpt_low, transformer_low, global_step, num_train_epochs,
                                precomputed_state=final_state_low, use_mem_eff=args.mem_eff_save)
            if train_high and transformer_high is not None:
                save_model_ckpt(ckpt_high, transformer_high, global_step, num_train_epochs,
                                precomputed_state=final_state_high, use_mem_eff=args.mem_eff_save)
            logger.info("model saved.")


def wan_finetune_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """WAN 2.2 full fine-tuning specific arguments."""
    parser.add_argument(
        "--train_models",
        type=str,
        choices=["low", "high", "both"],
        default=None,
        help="Which model(s) to train. 'low' = low-noise DiT (--dit), 'high' = high-noise DiT (--dit_high_noise),"
        " 'both' = train both. Default: both if both paths are provided, low/high if only one is provided."
        " / 学習するモデルを指定。'low'=低ノイズDiT、'high'=高ノイズDiT、'both'=両方。",
    )
    parser.add_argument(
        "--learning_rate_low",
        type=float,
        default=None,
        help="Learning rate for the low-noise model. Falls back to --learning_rate if not specified."
        " / 低ノイズモデルの学習率。未指定時は--learning_rateが使用されます。",
    )
    parser.add_argument(
        "--learning_rate_high",
        type=float,
        default=None,
        help="Learning rate for the high-noise model. Falls back to --learning_rate if not specified."
        " / 高ノイズモデルの学習率。未指定時は--learning_rateが使用されます。",
    )
    parser.add_argument(
        "--full_bf16",
        action="store_true",
        help="Enable full bfloat16 training (DiT weights in bf16). Requires --mixed_precision bf16."
        " / bfloat16でDiTを学習します。--mixed_precision bf16が必要です。",
    )
    parser.add_argument(
        "--mem_eff_save",
        action="store_true",
        help="Use memory-efficient checkpoint saving."
        " / メモリ効率の良いチェックポイント保存を使用します。",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = wan_setup_parser(parser)
    parser = wan_finetune_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None  # auto-detected in handle_model_specific_args

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = WanTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
