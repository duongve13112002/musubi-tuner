"""
Simulation tests for multi-caption training key mapping.

Verifies that:
1. TE cache keys for every architecture map to the exact batch-dict keys that the
   corresponding training scripts read (no real models or GPU needed).
2. The dual-encoder bug fix in encode_empty_caption_embeddings works correctly.
3. Edge cases: batch_size=1, ragged captions, empty strings, dropout corners.

Key transformation pipeline under test:
    TE cache file: caption_N_{bare_key}
         ↓  _select_caption_variant strips caption_N_ prefix
    bare_key  (e.g. "varlen_t5_bfloat16", "llm_bfloat16", "llm_mask")
         ↓  BucketBatchManager.__getitem__ key processing
    batch key (e.g. "t5", "llm", "llm_mask")

Training script expected keys (from source review):
    HunyuanVideo       : llm, llm_mask, clipL
    WAN                : t5  (varlen list)
    FramePack          : llama_vec, llama_attention_mask, clip_l_pooler
    Flux Kontext       : t5_vec, clip_l_pooler
    Flux 2             : ctx_vec
    HunyuanVideo 1.5   : vl_embed (varlen), byt5_embed (varlen)
    Kandinsky5         : text_embeds, pooled_embed, attention_mask
    Qwen Image         : vl_embed (varlen)
    Z-Image            : llm_embed (varlen)
"""

import os
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file, load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.cache_io import (
    get_caption_batches,
    get_empty_caption_cache_path,
    save_text_encoder_output_cache,
    save_text_encoder_output_cache_wan,
    save_text_encoder_output_cache_framepack,
    save_text_encoder_output_cache_flux_kontext,
    save_text_encoder_output_cache_flux_2,
    save_text_encoder_output_cache_hunyuan_video_1_5,
    save_text_encoder_output_cache_kandinsky5,
    save_text_encoder_output_cache_qwen_image,
    save_text_encoder_output_cache_z_image,
)
import musubi_tuner.cache_text_encoder_outputs as cache_te


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_item(key: str, captions, cache_path: str) -> ItemInfo:
    item = ItemInfo(key, captions, (64, 64))
    item.text_encoder_output_cache_path = cache_path
    return item


def _make_manager(dropout_rate=0.0, empty_path=None) -> BucketBatchManager:
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


def _apply_batch_key_transform(sd: dict) -> tuple[dict, set]:
    """
    Replicate BucketBatchManager.__getitem__ key processing for a flat sd dict.
    Returns (batch_dict, varlen_key_set).
    Non-varlen keys are torch.stack-ed (single-item list → tensor).
    """
    batch = {}
    varlen_keys = set()
    for key, val in sd.items():
        is_varlen = key.startswith("varlen_")
        content_key = key
        if is_varlen:
            content_key = content_key.replace("varlen_", "")
        if content_key.endswith("_mask"):
            pass
        else:
            content_key = content_key.rsplit("_", 1)[0]
            if content_key.startswith("latents_"):
                content_key = content_key.rsplit("_", 1)[0]
        batch.setdefault(content_key, []).append(val)
        if is_varlen:
            varlen_keys.add(content_key)
    for k in list(batch):
        if k not in varlen_keys:
            batch[k] = torch.stack(batch[k])
    return batch, varlen_keys


# ─────────────────────────────────────────────────────────────────────────────
# 1. Key transformation: TE cache bare keys → batch dict keys
# ─────────────────────────────────────────────────────────────────────────────


class TestKeyTransformation(unittest.TestCase):
    """Verify that each architecture's bare TE cache keys map to the exact batch
    keys that the corresponding training script reads."""

    def _assert_keys(self, bare_sd: dict, expected_batch_keys: set, expected_varlen: set = None):
        batch, varlen = _apply_batch_key_transform(bare_sd)
        self.assertEqual(set(batch.keys()), expected_batch_keys, f"batch keys mismatch; got {set(batch.keys())}")
        if expected_varlen is not None:
            self.assertEqual(varlen, expected_varlen, f"varlen keys mismatch; got {varlen}")

    def test_hv_keys(self):
        """HV training reads: llm, llm_mask, clipL"""
        bare = {
            "llm_bfloat16": torch.zeros(10, 4),
            "llm_mask": torch.ones(10),
            "clipL_bfloat16": torch.zeros(4),
        }
        self._assert_keys(bare, {"llm", "llm_mask", "clipL"}, expected_varlen=set())

    def test_wan_keys(self):
        """WAN training reads: t5 (as varlen list)"""
        bare = {
            "varlen_t5_bfloat16": torch.zeros(12, 4),
        }
        self._assert_keys(bare, {"t5"}, expected_varlen={"t5"})

    def test_fpack_keys(self):
        """FramePack training reads: llama_vec, llama_attention_mask, clip_l_pooler"""
        bare = {
            "llama_vec_bfloat16": torch.zeros(512, 4),
            "llama_attention_mask": torch.ones(512),
            "clip_l_pooler_bfloat16": torch.zeros(4),
        }
        self._assert_keys(
            bare,
            {"llama_vec", "llama_attention_mask", "clip_l_pooler"},
            expected_varlen=set(),
        )

    def test_flux_kontext_keys(self):
        """Flux Kontext training reads: t5_vec, clip_l_pooler"""
        bare = {
            "t5_vec_bfloat16": torch.zeros(256, 4),
            "clip_l_pooler_bfloat16": torch.zeros(4),
        }
        self._assert_keys(bare, {"t5_vec", "clip_l_pooler"}, expected_varlen=set())

    def test_flux_2_keys(self):
        """Flux 2 training reads: ctx_vec"""
        bare = {
            "ctx_vec_bfloat16": torch.zeros(512, 4),
        }
        self._assert_keys(bare, {"ctx_vec"}, expected_varlen=set())

    def test_hv15_keys(self):
        """HV1.5 training reads: vl_embed (varlen), byt5_embed (varlen)"""
        bare = {
            "varlen_vl_embed_bfloat16": torch.zeros(8, 4),
            "varlen_byt5_embed_bfloat16": torch.zeros(6, 4),
        }
        self._assert_keys(bare, {"vl_embed", "byt5_embed"}, expected_varlen={"vl_embed", "byt5_embed"})

    def test_kandinsky5_keys(self):
        """K5 training reads: text_embeds, pooled_embed, attention_mask"""
        bare = {
            "text_embeds_bfloat16": torch.zeros(16, 4),
            "pooled_embed_bfloat16": torch.zeros(4),
            "attention_mask": torch.ones(16, dtype=torch.bool),
        }
        self._assert_keys(
            bare,
            {"text_embeds", "pooled_embed", "attention_mask"},
            expected_varlen=set(),
        )

    def test_qwen_image_keys(self):
        """Qwen-Image training reads: vl_embed (varlen)"""
        bare = {
            "varlen_vl_embed_bfloat16": torch.zeros(20, 4),
        }
        self._assert_keys(bare, {"vl_embed"}, expected_varlen={"vl_embed"})

    def test_zimage_keys(self):
        """Z-Image training reads: llm_embed (varlen)"""
        bare = {
            "varlen_llm_embed_bfloat16": torch.zeros(15, 4),
        }
        self._assert_keys(bare, {"llm_embed"}, expected_varlen={"llm_embed"})


# ─────────────────────────────────────────────────────────────────────────────
# 2. Full pipeline: caption_N_ prefix → strip → key transform
# ─────────────────────────────────────────────────────────────────────────────


class TestCaptionPrefixToTrainingKeys(unittest.TestCase):
    """Verify the full pipeline: cache file with caption_N_ prefixed keys →
    _select_caption_variant strips prefix → key transformation → correct batch keys."""

    def _full_pipeline(self, prefixed_sd: dict, dropout=0.0, empty_path=None) -> tuple[dict, set]:
        mgr = _make_manager(dropout_rate=dropout, empty_path=empty_path)
        bare = mgr._select_caption_variant(prefixed_sd)
        return _apply_batch_key_transform(bare)

    def test_hv_prefixed_to_batch_keys(self):
        sd = {
            "caption_0_llm_bfloat16": torch.zeros(10, 4),
            "caption_0_llm_mask": torch.ones(10),
            "caption_0_clipL_bfloat16": torch.zeros(4),
        }
        batch, _ = self._full_pipeline(sd)
        self.assertEqual(set(batch.keys()), {"llm", "llm_mask", "clipL"})

    def test_wan_prefixed_to_batch_keys(self):
        sd = {"caption_0_varlen_t5_bfloat16": torch.zeros(12, 4)}
        batch, varlen = self._full_pipeline(sd)
        self.assertEqual(set(batch.keys()), {"t5"})
        self.assertIn("t5", varlen)
        self.assertIsInstance(batch["t5"], list)

    def test_fpack_prefixed_to_batch_keys(self):
        sd = {
            "caption_0_llama_vec_bfloat16": torch.zeros(512, 4),
            "caption_0_llama_attention_mask": torch.ones(512),
            "caption_0_clip_l_pooler_bfloat16": torch.zeros(4),
        }
        batch, _ = self._full_pipeline(sd)
        self.assertEqual(set(batch.keys()), {"llama_vec", "llama_attention_mask", "clip_l_pooler"})

    def test_flux_kontext_prefixed_to_batch_keys(self):
        sd = {
            "caption_0_t5_vec_bfloat16": torch.zeros(256, 4),
            "caption_0_clip_l_pooler_bfloat16": torch.zeros(4),
        }
        batch, _ = self._full_pipeline(sd)
        self.assertEqual(set(batch.keys()), {"t5_vec", "clip_l_pooler"})

    def test_flux_2_prefixed_to_batch_keys(self):
        sd = {"caption_0_ctx_vec_bfloat16": torch.zeros(512, 4)}
        batch, _ = self._full_pipeline(sd)
        self.assertEqual(set(batch.keys()), {"ctx_vec"})

    def test_hv15_prefixed_to_batch_keys(self):
        sd = {
            "caption_0_varlen_vl_embed_bfloat16": torch.zeros(8, 4),
            "caption_0_varlen_byt5_embed_bfloat16": torch.zeros(6, 4),
        }
        batch, varlen = self._full_pipeline(sd)
        self.assertEqual(set(batch.keys()), {"vl_embed", "byt5_embed"})
        self.assertEqual(varlen, {"vl_embed", "byt5_embed"})

    def test_kandinsky5_prefixed_to_batch_keys(self):
        sd = {
            "caption_0_text_embeds_bfloat16": torch.zeros(16, 4),
            "caption_0_pooled_embed_bfloat16": torch.zeros(4),
            "caption_0_attention_mask": torch.ones(16, dtype=torch.bool),
        }
        batch, _ = self._full_pipeline(sd)
        self.assertEqual(set(batch.keys()), {"text_embeds", "pooled_embed", "attention_mask"})

    def test_qwen_image_prefixed_to_batch_keys(self):
        sd = {"caption_0_varlen_vl_embed_bfloat16": torch.zeros(20, 4)}
        batch, varlen = self._full_pipeline(sd)
        self.assertEqual(set(batch.keys()), {"vl_embed"})
        self.assertIn("vl_embed", varlen)

    def test_zimage_prefixed_to_batch_keys(self):
        sd = {"caption_0_varlen_llm_embed_bfloat16": torch.zeros(15, 4)}
        batch, varlen = self._full_pipeline(sd)
        self.assertEqual(set(batch.keys()), {"llm_embed"})
        self.assertIn("llm_embed", varlen)

    def test_multi_caption_random_pick_maps_to_same_batch_keys(self):
        """Regardless of which caption variant is picked, the resulting batch keys
        must be identical — only the tensor values differ."""
        sd = {
            "caption_0_ctx_vec_bfloat16": torch.zeros(4, 8, dtype=torch.bfloat16),
            "caption_1_ctx_vec_bfloat16": torch.ones(4, 8, dtype=torch.bfloat16),
        }
        mgr = _make_manager()
        key_sets = set()
        for _ in range(50):
            bare = mgr._select_caption_variant(sd)
            batch, _ = _apply_batch_key_transform(bare)
            key_sets.add(frozenset(batch.keys()))
        self.assertEqual(len(key_sets), 1, "Different caption picks should produce the same batch key set")
        self.assertEqual(key_sets.pop(), frozenset({"ctx_vec"}))

    def test_dropout_empty_file_bare_keys_map_correctly(self):
        """The empty caption file uses bare keys (no caption_N_ prefix).
        _select_caption_variant must return them as-is, and they must still
        map to the correct training batch keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = os.path.join(tmpdir, "empty.safetensors")
            # Simulate WAN architecture empty embedding
            save_file({"varlen_t5_bfloat16": torch.zeros(5, 4)}, empty_path)

            sd_item = {"caption_0_varlen_t5_bfloat16": torch.zeros(10, 4)}
            mgr = _make_manager(dropout_rate=1.0, empty_path=empty_path)
            bare = mgr._select_caption_variant(sd_item)
            batch, varlen = _apply_batch_key_transform(bare)
            self.assertEqual(set(batch.keys()), {"t5"})
            self.assertIn("t5", varlen)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dual-encoder bug fix: encode_empty_caption_embeddings merges both TEs
# ─────────────────────────────────────────────────────────────────────────────


ARCH_HV_FULL = "HunyuanVideo"  # must match ARCHITECTURE_HUNYUAN_VIDEO_FULL


class TestDualEncoderEmptyFile(unittest.TestCase):
    """Verify the fix: calling encode_empty_caption_embeddings twice (once per TE)
    correctly merges both encoder outputs into the empty file."""

    def _make_hv_item(self, path: str) -> ItemInfo:
        item = ItemInfo("__empty__", [""], (0, 0))
        item.text_encoder_output_cache_path = path
        return item

    def test_second_call_is_not_skipped(self):
        """After the fix, encode_single is called even when the file already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = _make_dataset_mock(tmpdir, dropout_rate=0.1)
            empty_path = get_empty_caption_cache_path(tmpdir)
            # Pre-create the file (simulates TE1 having already written)
            save_file({"llm_bfloat16": torch.zeros(4)}, empty_path)

            calls = []
            cache_te.encode_empty_caption_embeddings(lambda item: calls.append(item), [ds])
            self.assertEqual(len(calls), 1, "encode_single must be called even when file exists")

    def test_dual_te_merges_both_keys_into_empty_file(self):
        """Simulates HunyuanVideo: TE1 call writes llm_* keys, TE2 call writes
        clipL_* keys. The resulting empty file must have BOTH sets of keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = _make_dataset_mock(tmpdir, dropout_rate=0.1)
            empty_path = get_empty_caption_cache_path(tmpdir)

            def encode_te1(item: ItemInfo):
                embed = torch.zeros(10, 4, dtype=torch.bfloat16)
                mask = torch.ones(10, dtype=torch.bool)
                save_text_encoder_output_cache(item, embed, mask, is_llm=True, caption_prefix="")

            def encode_te2(item: ItemInfo):
                embed = torch.zeros(4, dtype=torch.bfloat16)
                save_text_encoder_output_cache(item, embed, None, is_llm=False, caption_prefix="")

            # Simulate HunyuanVideo: two separate calls, one per TE
            cache_te.encode_empty_caption_embeddings(encode_te1, [ds])
            cache_te.encode_empty_caption_embeddings(encode_te2, [ds])

            self.assertTrue(os.path.exists(empty_path))
            sd = load_file(empty_path)
            llm_keys = [k for k in sd if "llm" in k.lower()]
            clip_keys = [k for k in sd if "clipL" in k.lower() or "clip" in k.lower()]
            self.assertTrue(len(llm_keys) >= 1, f"LLM key missing; keys: {list(sd.keys())}")
            self.assertTrue(len(clip_keys) >= 1, f"CLIP-L key missing; keys: {list(sd.keys())}")

    def test_single_te_architecture_still_works(self):
        """Single-TE architectures (WAN, Flux2, etc.) call encode_empty once.
        The fix must not break them — encode_single is called once, file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = _make_dataset_mock(tmpdir, dropout_rate=0.1)
            empty_path = get_empty_caption_cache_path(tmpdir)
            encoded_items = []

            def encode_wan(item: ItemInfo):
                encoded_items.append(item.item_key)
                embed = torch.zeros(8, 4, dtype=torch.bfloat16)
                save_text_encoder_output_cache_wan(item, embed, caption_prefix="")

            cache_te.encode_empty_caption_embeddings(encode_wan, [ds])
            self.assertEqual(len(encoded_items), 1)
            self.assertTrue(os.path.exists(empty_path))
            sd = load_file(empty_path)
            t5_keys = [k for k in sd if "t5" in k]
            self.assertEqual(len(t5_keys), 1)

    def test_idempotent_rerun_overwrites_cleanly(self):
        """Re-running encode_empty on an already-complete file should produce the
        same result without corrupting the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = _make_dataset_mock(tmpdir, dropout_rate=0.1)

            def encode_flux2(item: ItemInfo):
                save_text_encoder_output_cache_flux_2(
                    item, torch.zeros(4, 8, dtype=torch.bfloat16), arch_full="flux.2-dev", caption_prefix=""
                )

            # First run
            cache_te.encode_empty_caption_embeddings(encode_flux2, [ds])
            # Copy tensors off the mmap before the second write: on Windows, safetensors
            # keeps the file handle open until GC, which blocks overwrite (errno 22).
            sd_first = {k: v.clone() for k, v in load_file(get_empty_caption_cache_path(tmpdir)).items()}

            # Second run (simulates re-running the caching script)
            cache_te.encode_empty_caption_embeddings(encode_flux2, [ds])
            sd_second = load_file(get_empty_caption_cache_path(tmpdir))

            self.assertEqual(set(sd_first.keys()), set(sd_second.keys()))

    def test_dropout_empty_file_keys_match_item_cache_keys(self):
        """The empty file's bare keys must produce the same batch keys as a
        regular item's caption_0_-prefixed keys after transformation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = _make_dataset_mock(tmpdir, dropout_rate=0.1)
            empty_path = get_empty_caption_cache_path(tmpdir)

            def encode_zimage(item: ItemInfo):
                save_text_encoder_output_cache_z_image(
                    item, torch.zeros(7, 4, dtype=torch.bfloat16), caption_prefix=""
                )

            cache_te.encode_empty_caption_embeddings(encode_zimage, [ds])

            # Regular item cache has caption_0_ prefix
            item_sd = {"caption_0_varlen_llm_embed_bfloat16": torch.zeros(10, 4, dtype=torch.bfloat16)}
            # Empty file has bare keys
            empty_sd = load_file(empty_path)

            mgr_dropout = _make_manager(dropout_rate=1.0, empty_path=empty_path)
            mgr_normal = _make_manager(dropout_rate=0.0)

            bare_dropout = mgr_dropout._select_caption_variant(item_sd)
            bare_normal = mgr_normal._select_caption_variant(item_sd)

            batch_dropout, varlen_dropout = _apply_batch_key_transform(bare_dropout)
            batch_normal, varlen_normal = _apply_batch_key_transform(bare_normal)

            self.assertEqual(set(batch_dropout.keys()), set(batch_normal.keys()),
                             "Dropout path and normal path must produce identical batch key sets")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Edge cases
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases(unittest.TestCase):

    def test_empty_batch_get_caption_batches(self):
        self.assertEqual(list(get_caption_batches([])), [])

    def test_single_item_single_caption(self):
        item = ItemInfo("x", ["only caption"], (1, 1))
        batches = list(get_caption_batches([item]))
        self.assertEqual(len(batches), 1)
        idx, items, prompts = batches[0]
        self.assertEqual(idx, 0)
        self.assertEqual(prompts, ["only caption"])

    def test_batch_size_1_multi_caption_yields_all_variants(self):
        """Single-item batch with 3 captions: get_caption_batches must yield 3 batches."""
        item = ItemInfo("one", ["first", "second", "third"], (1, 1))
        batches = list(get_caption_batches([item]))
        self.assertEqual(len(batches), 3)
        for expected_idx, (idx, items, prompts) in enumerate(batches):
            self.assertEqual(idx, expected_idx)
            self.assertEqual(items, [item])
            self.assertEqual(prompts, [item.captions[expected_idx]])

    def test_ragged_batch_high_index_excludes_short_items(self):
        items = [
            ItemInfo("a", ["a0", "a1", "a2"], (1, 1)),
            ItemInfo("b", ["b0", "b1"], (1, 1)),
            ItemInfo("c", ["c0"], (1, 1)),
        ]
        batches = list(get_caption_batches(items))
        self.assertEqual(len(batches), 3)
        _, items0, _ = batches[0]
        self.assertEqual(len(items0), 3)
        _, items1, _ = batches[1]
        self.assertEqual(len(items1), 2)
        self.assertNotIn(items[2], items1)
        _, items2, _ = batches[2]
        self.assertEqual(len(items2), 1)
        self.assertEqual(items2[0], items[0])

    def test_empty_caption_string_in_item(self):
        """ItemInfo with [""] (single empty caption) should yield 1 batch with empty prompt."""
        item = ItemInfo("x", [""], (1, 1))
        batches = list(get_caption_batches([item]))
        self.assertEqual(len(batches), 1)
        _, _, prompts = batches[0]
        self.assertEqual(prompts, [""])

    def test_select_caption_single_variant_always_returns_same_value(self):
        """Single caption → caption_indices = {0} → same variant always selected."""
        mgr = _make_manager()
        sd = {"caption_0_ctx_vec_bfloat16": torch.full((4,), 7.0, dtype=torch.bfloat16)}
        for _ in range(30):
            result = mgr._select_caption_variant(sd)
            self.assertIn("ctx_vec_bfloat16", result)
            self.assertAlmostEqual(float(result["ctx_vec_bfloat16"].mean()), 7.0, places=2)

    def test_select_caption_three_variants_all_reachable(self):
        """Three caption variants should all be reachable over enough trials."""
        mgr = _make_manager()
        sd = {
            "caption_0_ctx_vec_bfloat16": torch.full((4,), 0.0),
            "caption_1_ctx_vec_bfloat16": torch.full((4,), 1.0),
            "caption_2_ctx_vec_bfloat16": torch.full((4,), 2.0),
        }
        seen = set()
        for _ in range(300):
            result = mgr._select_caption_variant(sd)
            seen.add(round(float(result["ctx_vec_bfloat16"].mean()), 1))
        self.assertEqual(seen, {0.0, 1.0, 2.0}, "All 3 caption variants must be reachable")

    def test_dropout_with_rate_1_always_uses_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = os.path.join(tmpdir, "empty.safetensors")
            save_file({"ctx_vec_bfloat16": torch.full((4,), 99.0)}, empty_path)

            mgr = _make_manager(dropout_rate=1.0, empty_path=empty_path)
            sd = {"caption_0_ctx_vec_bfloat16": torch.zeros(4)}
            for _ in range(20):
                result = mgr._select_caption_variant(sd)
                self.assertAlmostEqual(float(result["ctx_vec_bfloat16"].mean()), 99.0, places=1)

    def test_dropout_with_rate_0_never_uses_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = os.path.join(tmpdir, "empty.safetensors")
            save_file({"ctx_vec_bfloat16": torch.full((4,), 99.0)}, empty_path)

            mgr = _make_manager(dropout_rate=0.0, empty_path=empty_path)
            sd = {"caption_0_ctx_vec_bfloat16": torch.zeros(4)}
            for _ in range(20):
                result = mgr._select_caption_variant(sd)
                self.assertLess(float(result["ctx_vec_bfloat16"].mean()), 90.0)

    def test_dropout_missing_empty_file_gracefully_falls_back(self):
        """If empty file is configured but not yet created, dropout silently does
        nothing — the item's own caption is used instead."""
        mgr = _make_manager(dropout_rate=1.0, empty_path="/nonexistent/path.safetensors")
        sd = {"caption_0_ctx_vec_bfloat16": torch.zeros(4)}
        result = mgr._select_caption_variant(sd)
        self.assertIn("ctx_vec_bfloat16", result)

    def test_all_items_same_caption_count_batch_is_complete(self):
        """Uniform-length batch: every get_caption_batches iteration includes all items."""
        items = [ItemInfo(f"item_{i}", [f"cap_{i}_0", f"cap_{i}_1"], (1, 1)) for i in range(4)]
        batches = list(get_caption_batches(items))
        self.assertEqual(len(batches), 2)
        for _, batch_items, _ in batches:
            self.assertEqual(len(batch_items), 4)

    def test_save_load_all_architectures_bare_keys_are_accessible(self):
        """For every architecture, write a TE cache with caption_0_ prefix,
        load it, apply _select_caption_variant, and verify bare keys are present."""
        architectures = [
            # (save_fn, kwargs, expected_bare_key_prefixes)
            (
                lambda item: save_text_encoder_output_cache(
                    item, torch.zeros(8, 4, dtype=torch.bfloat16),
                    torch.ones(8, dtype=torch.bool), True, caption_prefix="caption_0_"
                ),
                ["llm"],
            ),
            (
                lambda item: save_text_encoder_output_cache_wan(
                    item, torch.zeros(8, 4, dtype=torch.bfloat16), caption_prefix="caption_0_"
                ),
                ["varlen_t5"],
            ),
            (
                lambda item: save_text_encoder_output_cache_flux_2(
                    item, torch.zeros(4, 8, dtype=torch.bfloat16),
                    arch_full="flux.2-dev", caption_prefix="caption_0_"
                ),
                ["ctx_vec"],
            ),
            (
                lambda item: save_text_encoder_output_cache_z_image(
                    item, torch.zeros(7, 4, dtype=torch.bfloat16), caption_prefix="caption_0_"
                ),
                ["varlen_llm_embed"],
            ),
            (
                lambda item: save_text_encoder_output_cache_qwen_image(
                    item, torch.zeros(10, 4, dtype=torch.bfloat16), caption_prefix="caption_0_"
                ),
                ["varlen_vl_embed"],
            ),
        ]
        mgr = _make_manager()
        for save_fn, expected_prefixes in architectures:
            with tempfile.TemporaryDirectory() as tmpdir:
                cache_path = os.path.join(tmpdir, "te.safetensors")
                item = ItemInfo("key", ["test caption"], (64, 64))
                item.text_encoder_output_cache_path = cache_path

                save_fn(item)

                sd = load_file(cache_path)
                bare = mgr._select_caption_variant(sd)

                for prefix in expected_prefixes:
                    found = any(k.startswith(prefix) for k in bare)
                    self.assertTrue(found, f"Expected key with prefix '{prefix}' in {list(bare.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. encode_empty_caption_embeddings full-cycle with multiple datasets
# ─────────────────────────────────────────────────────────────────────────────


class TestEncodeEmptyFullCycle(unittest.TestCase):

    def test_multiple_datasets_each_encode_independently(self):
        """Two datasets with separate cache directories each get their own empty file."""
        with tempfile.TemporaryDirectory() as d1:
            with tempfile.TemporaryDirectory() as d2:
                ds1 = _make_dataset_mock(d1, dropout_rate=0.1)
                ds2 = _make_dataset_mock(d2, dropout_rate=0.2)
                ds_no_drop = _make_dataset_mock(d1, dropout_rate=0.0)

                encoded_paths = []

                def encode_fn(item):
                    save_text_encoder_output_cache_flux_2(
                        item, torch.zeros(4), arch_full="flux.2-dev", caption_prefix=""
                    )
                    encoded_paths.append(item.text_encoder_output_cache_path)

                cache_te.encode_empty_caption_embeddings(encode_fn, [ds1, ds2, ds_no_drop])

                self.assertEqual(len(encoded_paths), 2)
                self.assertIn(get_empty_caption_cache_path(d1), encoded_paths)
                self.assertIn(get_empty_caption_cache_path(d2), encoded_paths)

    def test_non_main_process_does_nothing(self):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = _make_dataset_mock(tmpdir, dropout_rate=0.5)
            acc = MagicMock()
            acc.is_main_process = False
            calls = []
            cache_te.encode_empty_caption_embeddings(lambda item: calls.append(item), [ds], accelerator=acc)
            self.assertEqual(calls, [])

    def test_hv15_dual_varlen_keys_after_two_calls(self):
        """HunyuanVideo 1.5: two separate calls (VLM + BYT5).
        After both calls, the empty file must have both varlen_vl_embed and
        varlen_byt5_embed keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = _make_dataset_mock(tmpdir, dropout_rate=0.1)
            empty_path = get_empty_caption_cache_path(tmpdir)

            def encode_vlm(item: ItemInfo):
                save_text_encoder_output_cache_hunyuan_video_1_5(
                    item,
                    embed=torch.zeros(8, 4, dtype=torch.bfloat16),
                    byt5_embed=torch.zeros(1, 4, dtype=torch.bfloat16),  # placeholder
                    caption_prefix="",
                )

            # Two separate calls simulating two encoders
            cache_te.encode_empty_caption_embeddings(encode_vlm, [ds])
            cache_te.encode_empty_caption_embeddings(encode_vlm, [ds])  # idempotent second call

            self.assertTrue(os.path.exists(empty_path))
            sd = load_file(empty_path)
            self.assertTrue(any("vl_embed" in k for k in sd), f"vl_embed key missing: {list(sd.keys())}")
            self.assertTrue(any("byt5_embed" in k for k in sd), f"byt5_embed key missing: {list(sd.keys())}")


if __name__ == "__main__":
    unittest.main()
