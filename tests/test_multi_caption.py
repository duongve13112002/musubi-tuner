"""
Simulation tests for multi-caption feature.

No real models or GPU required. All encoding is either mocked or done with
random tensors. Tests verify correctness of:

1. get_caption_batches: yields (idx, items, prompts) correctly for ragged batches
2. _select_caption_variant: strips prefix, respects dropout, handles legacy format
3. encode_empty_caption_embeddings: main-process-only, skips if file exists
4. End-to-end save/merge/select cycle using safetensors files
5. Datasource caption reading: multi-line .txt and JSONL "captions" key
"""

import json
import os
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file, load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.dataset.cache_io import (
    get_caption_batches,
    get_empty_caption_cache_path,
    save_text_encoder_output_cache_flux_2,
)
from musubi_tuner.dataset.bucket import BucketBatchManager
import musubi_tuner.cache_text_encoder_outputs as cache_te


# Helpers


def _make_item(key: str, captions: list[str], cache_path: str) -> ItemInfo:
    item = ItemInfo(key, captions, (64, 64))
    item.text_encoder_output_cache_path = cache_path
    return item


def _make_manager(dropout_rate: float = 0.0, empty_path: str = None) -> BucketBatchManager:
    """Minimal BucketBatchManager just for testing _select_caption_variant."""
    mgr = BucketBatchManager.__new__(BucketBatchManager)
    mgr.caption_dropout_rate = dropout_rate
    mgr.empty_caption_cache_path = empty_path
    mgr.timestep_pool = None
    return mgr


def _make_dataset_mock(cache_dir: str, dropout_rate: float):
    from unittest.mock import MagicMock
    ds = MagicMock()
    ds.caption_dropout_rate = dropout_rate
    ds.cache_directory = cache_dir
    return ds


# 1. get_caption_batches


class TestGetCaptionBatches(unittest.TestCase):
    def test_single_caption_yields_one_batch(self):
        items = [ItemInfo(f"item_{i}", [f"cap{i}"], (1, 1)) for i in range(3)]
        batches = list(get_caption_batches(items))
        self.assertEqual(len(batches), 1)
        idx, batch_items, prompts = batches[0]
        self.assertEqual(idx, 0)
        self.assertEqual(batch_items, items)
        self.assertEqual(prompts, ["cap0", "cap1", "cap2"])

    def test_equal_length_multi_caption(self):
        items = [ItemInfo(f"x{i}", [f"a{i}", f"b{i}", f"c{i}"], (1, 1)) for i in range(2)]
        batches = list(get_caption_batches(items))
        self.assertEqual(len(batches), 3)
        for expected_idx, (idx, batch_items, prompts) in enumerate(batches):
            self.assertEqual(idx, expected_idx)
            self.assertEqual(len(batch_items), 2)
            for j in range(2):
                expected_cap = ["a", "b", "c"][expected_idx] + str(j)
                self.assertEqual(prompts[j], expected_cap)

    def test_ragged_captions_excludes_short_items(self):
        item_long = ItemInfo("long", ["L0", "L1", "L2"], (1, 1))
        item_short = ItemInfo("short", ["S0"], (1, 1))
        batches = list(get_caption_batches([item_long, item_short]))
        self.assertEqual(len(batches), 3)
        # idx 0: both items
        idx, items0, prompts0 = batches[0]
        self.assertEqual(idx, 0)
        self.assertIn(item_long, items0)
        self.assertIn(item_short, items0)
        self.assertCountEqual(prompts0, ["L0", "S0"])
        # idx 1 and 2: only item_long
        for idx, batch_items, prompts in batches[1:]:
            self.assertEqual(batch_items, [item_long])

    def test_empty_batch_yields_nothing(self):
        self.assertEqual(list(get_caption_batches([])), [])

    def test_prompts_match_caption_index(self):
        item = ItemInfo("item", ["first", "second", "third"], (1, 1))
        for idx, batch_items, prompts in get_caption_batches([item]):
            self.assertEqual(prompts, [item.captions[idx]])

    def test_caption_count_determines_max_index(self):
        items = [
            ItemInfo("a", ["a"], (1, 1)),
            ItemInfo("b", ["b0", "b1"], (1, 1)),
            ItemInfo("c", ["c0", "c1", "c2", "c3"], (1, 1)),
        ]
        indices = [idx for idx, _, _ in get_caption_batches(items)]
        self.assertEqual(indices, [0, 1, 2, 3])


# 2. _select_caption_variant


class TestSelectCaptionVariant(unittest.TestCase):
    def test_bare_keys_pass_through_unchanged(self):
        """Legacy single-caption format → returned as-is."""
        mgr = _make_manager()
        sd = {
            "embed_bfloat16": torch.zeros(4, 8),
            "mask": torch.ones(4),
        }
        result = mgr._select_caption_variant(sd)
        self.assertEqual(result, sd)

    def test_multi_caption_strips_prefix(self):
        mgr = _make_manager()
        sd = {
            "caption_0_embed_bfloat16": torch.zeros(4, 8),
            "caption_0_mask": torch.ones(4),
            "caption_1_embed_bfloat16": torch.ones(4, 8) * 2,
            "caption_1_mask": torch.zeros(4),
        }
        result = mgr._select_caption_variant(sd)
        self.assertIn("embed_bfloat16", result)
        self.assertIn("mask", result)
        self.assertNotIn("caption_0_embed_bfloat16", result)
        self.assertNotIn("caption_1_embed_bfloat16", result)

    def test_both_variants_are_reachable(self):
        """Over many calls, both caption variants should be selected at some point."""
        mgr = _make_manager()
        sd = {
            "caption_0_embed_bfloat16": torch.zeros(4),
            "caption_1_embed_bfloat16": torch.ones(4) * 2,
        }
        selected_values = set()
        for _ in range(200):
            result = mgr._select_caption_variant(sd)
            val = round(float(result["embed_bfloat16"].mean()), 1)
            selected_values.add(val)
        self.assertEqual(selected_values, {0.0, 2.0})

    def test_caption_dropout_with_rate_one_uses_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = os.path.join(tmpdir, "empty.safetensors")
            empty_embed = torch.full((3,), 99.0)
            save_file({"embed_bfloat16": empty_embed}, empty_path)

            mgr = _make_manager(dropout_rate=1.0, empty_path=empty_path)
            sd = {"caption_0_embed_bfloat16": torch.zeros(4)}
            result = mgr._select_caption_variant(sd)
            self.assertTrue(torch.allclose(result["embed_bfloat16"], empty_embed))

    def test_caption_dropout_missing_file_falls_back_to_caption(self):
        mgr = _make_manager(dropout_rate=1.0, empty_path="/nonexistent/empty.safetensors")
        sd = {
            "caption_0_embed_bfloat16": torch.zeros(4),
            "caption_1_embed_bfloat16": torch.ones(4),
        }
        result = mgr._select_caption_variant(sd)
        self.assertIn("embed_bfloat16", result)

    def test_zero_dropout_never_uses_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = os.path.join(tmpdir, "empty.safetensors")
            save_file({"embed_bfloat16": torch.full((3,), 99.0)}, empty_path)

            mgr = _make_manager(dropout_rate=0.0, empty_path=empty_path)
            sd = {"caption_0_embed_bfloat16": torch.zeros(4)}
            for _ in range(50):
                result = mgr._select_caption_variant(sd)
                self.assertFalse(float(result["embed_bfloat16"].mean()) > 90)


# 3. encode_empty_caption_embeddings


class TestEncodeEmptyCaptionEmbeddings(unittest.TestCase):
    def test_skips_dataset_with_no_dropout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = _make_dataset_mock(tmpdir, dropout_rate=0.0)
            calls = []
            cache_te.encode_empty_caption_embeddings(lambda item: calls.append(item), [ds])
            self.assertEqual(calls, [])

    def test_encodes_once_for_dropout_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = _make_dataset_mock(tmpdir, dropout_rate=0.1)
            encoded = []
            cache_te.encode_empty_caption_embeddings(lambda item: encoded.append(item), [ds])
            self.assertEqual(len(encoded), 1)
            self.assertEqual(encoded[0].captions, [""])
            self.assertEqual(
                encoded[0].text_encoder_output_cache_path,
                get_empty_caption_cache_path(tmpdir),
            )

    def test_always_calls_encode_even_if_file_exists(self):
        # Regression test for dual-encoder bug: HunyuanVideo calls encode_empty_caption_embeddings
        # twice (once per TE). The old early-exit on os.path.exists caused TE2 to be skipped,
        # leaving the empty file with only TE1 keys. merge-on-save handles idempotency correctly.
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = _make_dataset_mock(tmpdir, dropout_rate=0.1)
            empty_path = get_empty_caption_cache_path(tmpdir)
            save_file({"embed_bfloat16": torch.zeros(4)}, empty_path)
            calls = []
            cache_te.encode_empty_caption_embeddings(lambda item: calls.append(item), [ds])
            self.assertEqual(len(calls), 1)

    def test_non_main_process_does_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from unittest.mock import MagicMock
            ds = _make_dataset_mock(tmpdir, dropout_rate=0.1)
            acc = MagicMock()
            acc.is_main_process = False
            calls = []
            cache_te.encode_empty_caption_embeddings(lambda item: calls.append(item), [ds], accelerator=acc)
            self.assertEqual(calls, [])

    def test_multiple_datasets_each_get_separate_file(self):
        with tempfile.TemporaryDirectory() as d1:
            with tempfile.TemporaryDirectory() as d2:
                ds1 = _make_dataset_mock(d1, dropout_rate=0.1)
                ds2 = _make_dataset_mock(d2, dropout_rate=0.2)
                ds_no_drop = _make_dataset_mock(d1, dropout_rate=0.0)
                paths = []
                cache_te.encode_empty_caption_embeddings(
                    lambda item: paths.append(item.text_encoder_output_cache_path),
                    [ds1, ds2, ds_no_drop],
                )
                self.assertEqual(len(paths), 2)
                self.assertIn(get_empty_caption_cache_path(d1), paths)
                self.assertIn(get_empty_caption_cache_path(d2), paths)


# 4. End-to-end: save multiple captions → merge → select


ARCH_FLUX2_FULL = "flux.2-dev"  # any non-empty string works for testing


class TestMultiCaptionCacheEndToEnd(unittest.TestCase):
    def _make_item_with_captions(self, cache_path: str, captions: list[str]) -> ItemInfo:
        item = ItemInfo("test_item", captions, (512, 512))
        item.text_encoder_output_cache_path = cache_path
        return item

    def test_save_two_captions_merges_into_one_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "te_cache.safetensors")
            item = self._make_item_with_captions(cache_path, ["caption A", "caption B"])

            embed0 = torch.zeros(8, 16, dtype=torch.bfloat16)
            embed1 = torch.ones(8, 16, dtype=torch.bfloat16)

            save_text_encoder_output_cache_flux_2(item, embed0, arch_full=ARCH_FLUX2_FULL, caption_prefix="caption_0_")
            save_text_encoder_output_cache_flux_2(item, embed1, arch_full=ARCH_FLUX2_FULL, caption_prefix="caption_1_")

            sd = load_file(cache_path)
            self.assertIn("caption_0_ctx_vec_bfloat16", sd)
            self.assertIn("caption_1_ctx_vec_bfloat16", sd)
            self.assertEqual(len([k for k in sd if k.startswith("caption_")]), 2)

    def test_select_caption_variant_returns_one_caption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "te_cache.safetensors")
            item = self._make_item_with_captions(cache_path, ["A", "B"])

            embed0 = torch.zeros(8, 16, dtype=torch.bfloat16)
            embed1 = torch.full((8, 16), 2.0, dtype=torch.bfloat16)

            save_text_encoder_output_cache_flux_2(item, embed0, arch_full=ARCH_FLUX2_FULL, caption_prefix="caption_0_")
            save_text_encoder_output_cache_flux_2(item, embed1, arch_full=ARCH_FLUX2_FULL, caption_prefix="caption_1_")

            sd = load_file(cache_path)
            mgr = _make_manager()

            results = set()
            for _ in range(100):
                result = mgr._select_caption_variant(sd)
                self.assertIn("ctx_vec_bfloat16", result)
                self.assertNotIn("caption_0_ctx_vec_bfloat16", result)
                val = round(float(result["ctx_vec_bfloat16"].mean()), 1)
                results.add(val)
            # Both variants should be selected over 100 trials
            self.assertEqual(results, {0.0, 2.0})

    def test_caption_dropout_uses_empty_file_not_item_captions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "te_cache.safetensors")
            item = self._make_item_with_captions(cache_path, ["A"])
            embed = torch.zeros(8, 16, dtype=torch.bfloat16)
            save_text_encoder_output_cache_flux_2(item, embed, arch_full=ARCH_FLUX2_FULL, caption_prefix="caption_0_")

            # Create empty embedding file with a unique sentinel value
            empty_path = os.path.join(tmpdir, "empty_caption_embeddings.safetensors")
            empty_embed = torch.full((4, 16), 42.0, dtype=torch.bfloat16)
            save_file({"ctx_vec_bfloat16": empty_embed}, empty_path)

            sd = load_file(cache_path)
            mgr = _make_manager(dropout_rate=1.0, empty_path=empty_path)
            result = mgr._select_caption_variant(sd)
            self.assertTrue(torch.allclose(result["ctx_vec_bfloat16"], empty_embed))

    def test_backward_compat_bare_keys_unchanged(self):
        """Files with bare keys (no caption_0_ prefix) pass through _select_caption_variant unchanged."""
        mgr = _make_manager()
        bare_sd = {
            "ctx_vec_bfloat16": torch.zeros(8, 16, dtype=torch.bfloat16),
        }
        result = mgr._select_caption_variant(bare_sd)
        self.assertIs(result, bare_sd)

    def test_single_item_encode_each_caption_index(self):
        """get_caption_batches with a 3-caption item yields all 3 caption embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "te_cache.safetensors")
            captions = ["first caption", "second caption", "third caption"]
            item = self._make_item_with_captions(cache_path, captions)

            for caption_idx, items, prompts in get_caption_batches([item]):
                prefix = f"caption_{caption_idx}_"
                embed = torch.full((4, 8), float(caption_idx), dtype=torch.bfloat16)
                self.assertEqual(prompts, [captions[caption_idx]])
                save_text_encoder_output_cache_flux_2(item, embed, arch_full=ARCH_FLUX2_FULL, caption_prefix=prefix)

            sd = load_file(cache_path)
            for i in range(3):
                key = f"caption_{i}_ctx_vec_bfloat16"
                self.assertIn(key, sd)
                self.assertAlmostEqual(float(sd[key].mean()), float(i), places=3)


# 5. Datasource caption reading


class TestDatasourceCaptionReading(unittest.TestCase):
    def _create_image(self, path: str):
        from PIL import Image
        img = Image.new("RGB", (4, 4))
        img.save(path)

    def test_image_directory_single_line_caption(self):
        from musubi_tuner.dataset.datasources import ImageDirectoryDatasource
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img.jpg")
            self._create_image(img_path)
            txt_path = os.path.join(tmpdir, "img.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("a single caption\n")

            ds = ImageDirectoryDatasource(tmpdir, caption_extension=".txt")
            key, captions = ds.get_caption(0)
            self.assertEqual(captions, ["a single caption"])

    def test_image_directory_multi_line_caption(self):
        from musubi_tuner.dataset.datasources import ImageDirectoryDatasource
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img.jpg")
            self._create_image(img_path)
            txt_path = os.path.join(tmpdir, "img.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("first line\nsecond line\nthird line\n")

            ds = ImageDirectoryDatasource(tmpdir, caption_extension=".txt")
            key, captions = ds.get_caption(0)
            self.assertEqual(captions, ["first line", "second line", "third line"])

    def test_image_directory_empty_lines_skipped(self):
        from musubi_tuner.dataset.datasources import ImageDirectoryDatasource
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img.jpg")
            self._create_image(img_path)
            txt_path = os.path.join(tmpdir, "img.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("caption one\n\n  \ncaption two\n")

            ds = ImageDirectoryDatasource(tmpdir, caption_extension=".txt")
            key, captions = ds.get_caption(0)
            self.assertEqual(captions, ["caption one", "caption two"])

    def test_image_directory_all_empty_returns_single_empty_string(self):
        from musubi_tuner.dataset.datasources import ImageDirectoryDatasource
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img.jpg")
            self._create_image(img_path)
            txt_path = os.path.join(tmpdir, "img.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("   \n  \n")

            ds = ImageDirectoryDatasource(tmpdir, caption_extension=".txt")
            key, captions = ds.get_caption(0)
            self.assertEqual(captions, [""])

    def test_jsonl_captions_key(self):
        from musubi_tuner.dataset.datasources import ImageJsonlDatasource
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img.jpg")
            self._create_image(img_path)
            jsonl_path = os.path.join(tmpdir, "data.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                json.dump({"image_path": img_path, "captions": ["caption A", "caption B", "caption C"]}, f)
                f.write("\n")

            ds = ImageJsonlDatasource(jsonl_path)
            key, captions = ds.get_caption(0)
            self.assertEqual(captions, ["caption A", "caption B", "caption C"])

    def test_jsonl_single_caption_backward_compat(self):
        """Old JSONL with single 'caption' key → wrapped in list."""
        from musubi_tuner.dataset.datasources import ImageJsonlDatasource
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img.jpg")
            self._create_image(img_path)
            jsonl_path = os.path.join(tmpdir, "data.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                json.dump({"image_path": img_path, "caption": "old single caption"}, f)
                f.write("\n")

            ds = ImageJsonlDatasource(jsonl_path)
            key, captions = ds.get_caption(0)
            self.assertEqual(captions, ["old single caption"])

    def test_jsonl_empty_captions_returns_single_empty_string(self):
        from musubi_tuner.dataset.datasources import ImageJsonlDatasource
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img.jpg")
            self._create_image(img_path)
            jsonl_path = os.path.join(tmpdir, "data.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                json.dump({"image_path": img_path, "captions": ["", "   "]}, f)
                f.write("\n")

            ds = ImageJsonlDatasource(jsonl_path)
            key, captions = ds.get_caption(0)
            self.assertEqual(captions, [""])


# 6. ItemInfo caption property (backward compatibility)


class TestItemInfoCaptionProperty(unittest.TestCase):
    def test_caption_property_returns_first(self):
        item = ItemInfo("k", ["first", "second", "third"], (1, 1))
        self.assertEqual(item.caption, "first")

    def test_caption_setter_updates_first(self):
        item = ItemInfo("k", ["original", "second"], (1, 1))
        item.caption = "updated"
        self.assertEqual(item.captions[0], "updated")
        self.assertEqual(item.captions[1], "second")

    def test_single_string_constructor(self):
        item = ItemInfo("k", "single caption", (1, 1))
        self.assertEqual(item.captions, ["single caption"])
        self.assertEqual(item.caption, "single caption")

    def test_empty_list_defaults_to_empty_string(self):
        item = ItemInfo("k", [], (1, 1))
        self.assertEqual(item.captions, [""])
        self.assertEqual(item.caption, "")


if __name__ == "__main__":
    unittest.main()
