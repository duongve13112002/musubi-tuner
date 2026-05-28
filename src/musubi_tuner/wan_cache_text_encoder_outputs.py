import argparse
from typing import Optional

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
import accelerate

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_WAN, ItemInfo, save_text_encoder_output_cache_wan

# for t5 config: all Wan2.1 models have the same config for t5
from musubi_tuner.wan.configs import wan_t2v_14B

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
import logging

from musubi_tuner.wan.modules.t5 import T5EncoderModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(
    text_encoder: T5EncoderModel, batch: list[ItemInfo], device: torch.device, accelerator: Optional[accelerate.Accelerator]
):
    from musubi_tuner.dataset.cache_io import get_caption_batches
    for caption_idx, items, prompts in get_caption_batches(batch):
        caption_prefix = f"caption_{caption_idx}_"
        with torch.no_grad():
            if accelerator is not None:
                with accelerator.autocast():
                    context = text_encoder(prompts, device)
            else:
                context = text_encoder(prompts, device)
        for item, ctx in zip(items, context):
            save_text_encoder_output_cache_wan(item, ctx, caption_prefix=caption_prefix)


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = wan_setup_parser(parser)

    args = parser.parse_args()

    config = wan_t2v_14B.t2v_14B  # all Wan2.1 models have the same config for t5
    mixed_precision = ("bf16" if config.t5_dtype == torch.bfloat16 else "fp16") if args.fp8_t5 else "no"
    accelerator = accelerate.Accelerator(mixed_precision=mixed_precision)
    device = torch.device(args.device) if (args.device is not None and accelerator.num_processes == 1) else accelerator.device

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_WAN)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    # prepare cache files and paths: all_cache_files_for_dataset = exisiting cache files, all_cache_paths_for_dataset = all cache paths in the dataset
    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    # Load T5
    logger.info(f"Loading T5: {args.t5}")
    text_encoder = T5EncoderModel(
        text_len=config.text_len, dtype=config.t5_dtype, device=device, weight_path=args.t5, fp8=args.fp8_t5
    )

    # Encode with T5
    logger.info("Encoding with T5")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal text_encoder, device
        encode_and_save_batch(text_encoder, batch, device, accelerator if args.fp8_t5 else None)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
        accelerator=accelerator,
    )

    def encode_empty_wan(item: ItemInfo):
        nonlocal text_encoder
        with torch.no_grad():
            if args.fp8_t5:
                with accelerator.autocast():
                    context = text_encoder([item.caption], device)
            else:
                context = text_encoder([item.caption], device)
        save_text_encoder_output_cache_wan(item, context[0], caption_prefix="")

    cache_text_encoder_outputs.encode_empty_caption_embeddings(encode_empty_wan, datasets, accelerator)
    del text_encoder

    # remove cache files not in dataset
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache, accelerator=accelerator
    )


def wan_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--t5", type=str, default=None, required=True, help="text encoder (T5) checkpoint path")
    parser.add_argument("--fp8_t5", action="store_true", help="use fp8 for Text Encoder model")
    return parser


if __name__ == "__main__":
    main()
