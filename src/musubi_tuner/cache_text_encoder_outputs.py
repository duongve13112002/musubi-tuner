import argparse
import os
from typing import Optional, Union

import torch
from tqdm import tqdm

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
import accelerate

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_HUNYUAN_VIDEO,
    BaseDataset,
    ItemInfo,
    save_text_encoder_output_cache,
)
from musubi_tuner.hunyuan_model import text_encoder as text_encoder_module
from musubi_tuner.hunyuan_model.text_encoder import TextEncoder

import logging

from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_prompt(text_encoder: TextEncoder, prompt: Union[str, list[str]]):
    data_type = "video"  # video only, image is not supported
    text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

    with torch.no_grad():
        prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type)

    return prompt_outputs.hidden_state, prompt_outputs.attention_mask


def encode_and_save_batch(
    text_encoder: TextEncoder, batch: list[ItemInfo], is_llm: bool, accelerator: Optional[accelerate.Accelerator]
):
    prompts = [item.caption for item in batch]
    # print(prompts)

    # encode prompt
    if accelerator is not None:
        with accelerator.autocast():
            prompt_embeds, prompt_mask = encode_prompt(text_encoder, prompts)
    else:
        prompt_embeds, prompt_mask = encode_prompt(text_encoder, prompts)

    # # convert to fp16 if needed
    # if prompt_embeds.dtype == torch.float32 and text_encoder.dtype != torch.float32:
    #     prompt_embeds = prompt_embeds.to(text_encoder.dtype)

    # save prompt cache
    for item, embed, mask in zip(batch, prompt_embeds, prompt_mask):
        save_text_encoder_output_cache(item, embed, mask, is_llm)


def prepare_cache_files_and_paths(datasets: list[BaseDataset]):
    all_cache_files_for_dataset = []  # exisiting cache files
    all_cache_paths_for_dataset = []  # all cache paths in the dataset
    for dataset in datasets:
        all_cache_files = [os.path.normpath(file) for file in dataset.get_all_text_encoder_output_cache_files()]
        all_cache_files = set(all_cache_files)
        all_cache_files_for_dataset.append(all_cache_files)

        all_cache_paths_for_dataset.append(set())
    return all_cache_files_for_dataset, all_cache_paths_for_dataset


def process_text_encoder_batches(
    num_workers: Optional[int],
    skip_existing: bool,
    batch_size: int,
    datasets: list[BaseDataset],
    all_cache_files_for_dataset: list[set],
    all_cache_paths_for_dataset: list[set],
    encode: callable,
    requires_content: Optional[bool] = False,
    accelerator: Optional[accelerate.Accelerator] = None,
):
    """
    Architecture independent processing of text encoder batches.
    """
    num_processes = accelerator.num_processes if accelerator is not None else 1
    process_index = accelerator.process_index if accelerator is not None else 0
    is_main_process = accelerator.is_main_process if accelerator is not None else True

    num_workers = num_workers if num_workers is not None else max(1, os.cpu_count() - 1)
    for dataset_idx, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{dataset_idx}]")
        all_cache_files = all_cache_files_for_dataset[dataset_idx]
        all_cache_paths = all_cache_paths_for_dataset[dataset_idx]

        if not requires_content:
            batches = dataset.retrieve_text_encoder_output_cache_batches(num_workers)  # return captions only
        else:
            batches = dataset.retrieve_latent_cache_batches(num_workers)  # return captions and images/videos

        global_item_index = 0
        for batch in tqdm(batches, disable=not is_main_process):
            # update cache paths on every process for correct cleanup tracking
            if requires_content:
                batch = batch[1]  # batch is (key, items), so use items
            all_cache_paths.update([os.path.normpath(item.text_encoder_output_cache_path) for item in batch])

            # shard: each process encodes only its assigned items
            if num_processes > 1:
                shard_batch = [item for j, item in enumerate(batch) if (global_item_index + j) % num_processes == process_index]
            else:
                shard_batch = batch
            global_item_index += len(batch)

            # skip existing cache files
            if skip_existing:
                shard_batch = [
                    item for item in shard_batch if os.path.normpath(item.text_encoder_output_cache_path) not in all_cache_files
                ]
                if len(shard_batch) == 0:
                    continue

            bs = batch_size if batch_size is not None else len(shard_batch)
            for j in range(0, len(shard_batch), bs):
                encode(shard_batch[j : j + bs])


def post_process_cache_files(
    datasets: list[BaseDataset],
    all_cache_files_for_dataset: list[set],
    all_cache_paths_for_dataset: list[set],
    keep_cache: bool,
    accelerator: Optional[accelerate.Accelerator] = None,
):
    # barrier: ensure all processes finish writing before any cleanup
    if accelerator is not None:
        accelerator.wait_for_everyone()

    # only the main process removes stale cache files
    if accelerator is not None and not accelerator.is_main_process:
        return

    for i, dataset in enumerate(datasets):
        all_cache_files = all_cache_files_for_dataset[i]
        all_cache_paths = all_cache_paths_for_dataset[i]
        for cache_file in all_cache_files:
            if cache_file not in all_cache_paths:
                if keep_cache:
                    logger.info(f"Keep cache file not in the dataset: {cache_file}")
                else:
                    os.remove(cache_file)
                    logger.info(f"Removed old cache file: {cache_file}")


def main():
    parser = setup_parser_common()
    parser = hv_setup_parser(parser)

    args = parser.parse_args()

    accelerator = accelerate.Accelerator(mixed_precision="fp16" if args.fp8_llm else "no")
    device = torch.device(args.device) if (args.device is not None and accelerator.num_processes == 1) else accelerator.device

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_HUNYUAN_VIDEO)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    # prepare cache files and paths: all_cache_files_for_dataset = exisiting cache files, all_cache_paths_for_dataset = all cache paths in the dataset
    all_cache_files_for_dataset, all_cache_paths_for_dataset = prepare_cache_files_and_paths(datasets)

    # Load Text Encoder 1
    text_encoder_dtype = torch.float16 if args.text_encoder_dtype is None else str_to_dtype(args.text_encoder_dtype)
    logger.info(f"loading text encoder 1: {args.text_encoder1}")
    text_encoder_1 = text_encoder_module.load_text_encoder_1(args.text_encoder1, device, args.fp8_llm, text_encoder_dtype)
    text_encoder_1.to(device=device)

    # Encode with Text Encoder 1 (LLM)
    logger.info("Encoding with Text Encoder 1")

    def encode_for_text_encoder_1(batch: list[ItemInfo]):
        nonlocal text_encoder_1
        encode_and_save_batch(text_encoder_1, batch, is_llm=True, accelerator=accelerator)

    process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder_1,
        accelerator=accelerator,
    )
    del text_encoder_1

    # Load Text Encoder 2
    logger.info(f"loading text encoder 2: {args.text_encoder2}")
    text_encoder_2 = text_encoder_module.load_text_encoder_2(args.text_encoder2, device, text_encoder_dtype)
    text_encoder_2.to(device=device)

    # Encode with Text Encoder 2
    logger.info("Encoding with Text Encoder 2")

    def encode_for_text_encoder_2(batch: list[ItemInfo]):
        nonlocal text_encoder_2
        encode_and_save_batch(text_encoder_2, batch, is_llm=False, accelerator=None)

    process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder_2,
        accelerator=accelerator,
    )
    del text_encoder_2

    # remove cache files not in dataset
    post_process_cache_files(datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache, accelerator=accelerator)


def setup_parser_common():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_config", type=str, required=True, help="path to dataset config .toml file")
    parser.add_argument("--device", type=str, default=None, help="device to use, default is cuda if available")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="batch size, override dataset config if dataset batch size > this"
    )
    parser.add_argument("--num_workers", type=int, default=None, help="number of workers for dataset. default is cpu count-1")
    parser.add_argument("--skip_existing", action="store_true", help="skip existing cache files")
    parser.add_argument("--keep_cache", action="store_true", help="keep cache files not in dataset")
    return parser


def hv_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder1", type=str, required=True, help="Text Encoder 1 directory")
    parser.add_argument("--text_encoder2", type=str, required=True, help="Text Encoder 2 directory")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="data type for Text Encoder, default is float16")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for Text Encoder 1 (LLM)")
    return parser


if __name__ == "__main__":
    main()
