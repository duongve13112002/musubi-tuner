import argparse

import accelerate
import torch
from transformers import LlamaTokenizerFast, LlamaModel, CLIPTokenizer, CLIPTextModel
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_FRAMEPACK, ItemInfo, save_text_encoder_output_cache_framepack
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.frame_pack import hunyuan
from musubi_tuner.frame_pack.framepack_utils import load_text_encoder1, load_text_encoder2

import logging

from musubi_tuner.frame_pack.utils import crop_or_pad_yield_mask

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(
    tokenizer1: LlamaTokenizerFast,
    text_encoder1: LlamaModel,
    tokenizer2: CLIPTokenizer,
    text_encoder2: CLIPTextModel,
    batch: list[ItemInfo],
    device: torch.device,
):
    from musubi_tuner.dataset.cache_io import get_caption_batches
    # FramePack's encode_prompt_conds only supports single prompt, so we encode each prompt separately
    for caption_idx, items, prompts in get_caption_batches(batch):
        caption_prefix = f"caption_{caption_idx}_"
        list_of_llama_vec = []
        list_of_llama_attention_mask = []
        list_of_clip_l_pooler = []
        for prompt in prompts:
            with torch.autocast(device_type=device.type, dtype=text_encoder1.dtype), torch.no_grad():
                llama_vec, clip_l_pooler = hunyuan.encode_prompt_conds(prompt, text_encoder1, text_encoder2, tokenizer1, tokenizer2)
                llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            list_of_llama_vec.append(llama_vec.squeeze(0))
            list_of_llama_attention_mask.append(llama_attention_mask.squeeze(0))
            list_of_clip_l_pooler.append(clip_l_pooler.squeeze(0))
        for item, llama_vec, llama_attention_mask, clip_l_pooler in zip(
            items, list_of_llama_vec, list_of_llama_attention_mask, list_of_clip_l_pooler
        ):
            save_text_encoder_output_cache_framepack(item, llama_vec, llama_attention_mask, clip_l_pooler, caption_prefix=caption_prefix)


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = framepack_setup_parser(parser)

    args = parser.parse_args()

    accelerator = accelerate.Accelerator()
    device = torch.device(args.device) if (args.device is not None and accelerator.num_processes == 1) else accelerator.device

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_FRAMEPACK)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    # prepare cache files and paths: all_cache_files_for_dataset = exisiting cache files, all_cache_paths_for_dataset = all cache paths in the dataset
    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    # load text encoder
    tokenizer1, text_encoder1 = load_text_encoder1(args, args.fp8_llm, device)
    tokenizer2, text_encoder2 = load_text_encoder2(args)
    text_encoder2.to(device)

    # Encode with Text Encoders
    logger.info("Encoding with Text Encoders")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        encode_and_save_batch(tokenizer1, text_encoder1, tokenizer2, text_encoder2, batch, device)

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

    def encode_empty_fpack(item: ItemInfo):
        nonlocal tokenizer1, text_encoder1, tokenizer2, text_encoder2
        with torch.autocast(device_type=device.type, dtype=text_encoder1.dtype), torch.no_grad():
            llama_vec, clip_l_pooler = hunyuan.encode_prompt_conds(item.caption, text_encoder1, text_encoder2, tokenizer1, tokenizer2)
            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        save_text_encoder_output_cache_framepack(item, llama_vec.squeeze(0), llama_attention_mask.squeeze(0), clip_l_pooler.squeeze(0), caption_prefix="")

    cache_text_encoder_outputs.encode_empty_caption_embeddings(encode_empty_fpack, datasets, accelerator)

    # remove cache files not in dataset
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache, accelerator=accelerator
    )


def framepack_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder1", type=str, required=True, help="Text Encoder 1 directory")
    parser.add_argument("--text_encoder2", type=str, required=True, help="Text Encoder 2 directory")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for Text Encoder 1 (LLM)")
    return parser


if __name__ == "__main__":
    main()
