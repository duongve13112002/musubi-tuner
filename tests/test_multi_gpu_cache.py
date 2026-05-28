"""
Simulation tests for multi-GPU sharding in cache scripts.

These tests mock everything (accelerator, dataset, encode functions) so they run
without actual models or GPU hardware. They verify:

1. Data sharding: items are partitioned correctly across processes
2. No missed items: union of all shard assignments == full dataset
3. No duplicates: each item assigned to exactly one process
4. Cleanup gated on main process only
5. wait_for_everyone() barrier called before cleanup
6. tqdm disabled on non-main processes
7. Single-process fallback (no sharding) works identically to old behavior
"""

import argparse
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

import musubi_tuner.cache_latents as cache_latents
import musubi_tuner.cache_text_encoder_outputs as cache_te


# Helpers


def _make_item(idx: int, cache_dir: str, ext: str = ".safetensors") -> SimpleNamespace:
    """Create a lightweight fake ItemInfo for one dataset item."""
    item = SimpleNamespace()
    item.item_key = f"item_{idx:04d}"
    item.latent_cache_path = os.path.join(cache_dir, f"latent_{idx:04d}{ext}")
    item.text_encoder_output_cache_path = os.path.join(cache_dir, f"te_{idx:04d}{ext}")
    item.content = np.zeros((64, 64, 3), dtype=np.uint8)
    item.control_content = None
    item.frame_count = None
    return item


def _make_accelerator(num_processes: int, process_index: int) -> MagicMock:
    acc = MagicMock()
    acc.num_processes = num_processes
    acc.process_index = process_index
    acc.is_main_process = process_index == 0
    acc.wait_for_everyone = MagicMock()
    return acc


def _make_dataset(items: list, cache_dir: str) -> MagicMock:
    """Fake dataset that yields items in buckets of `bucket_size`."""
    ds = MagicMock()

    # retrieve_latent_cache_batches yields (key, batch) pairs
    def latent_batches(num_workers=None):
        # yield items one at a time as single-item batches
        for item in items:
            yield ("bucket_key", [item])

    ds.retrieve_latent_cache_batches = latent_batches

    # retrieve_text_encoder_output_cache_batches yields batches (no key)
    def te_batches(num_workers=None):
        for item in items:
            yield [item]

    ds.retrieve_text_encoder_output_cache_batches = te_batches

    # No stale cache files by default
    ds.get_all_latent_cache_files = lambda: []
    ds.get_all_text_encoder_output_cache_files = lambda: []

    return ds


def _make_args(cache_dir: str, skip_existing: bool = False) -> argparse.Namespace:
    args = argparse.Namespace()
    args.num_workers = 1
    args.skip_existing = skip_existing
    args.batch_size = 1
    args.keep_cache = False
    args.device = None
    return args


# Tests for cache_latents.encode_datasets


class TestEncodeDatasetsSingleProcess(unittest.TestCase):
    """Without accelerator, all items are encoded (backward-compat)."""

    def test_all_items_encoded(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            items = [_make_item(i, cache_dir) for i in range(8)]
            dataset = _make_dataset(items, cache_dir)
            args = _make_args(cache_dir)

            encoded = []

            def encode(batch):
                encoded.extend(batch)

            cache_latents.encode_datasets([dataset], encode, args, accelerator=None)

            self.assertEqual(len(encoded), 8, "All 8 items must be encoded in single-process mode")


class TestEncodeDatasetsMultiGPU(unittest.TestCase):
    def _simulate_all_processes(self, num_processes: int, num_items: int):
        """Simulate all N processes and return the union of encoded items."""
        all_encoded = []
        for pid in range(num_processes):
            with tempfile.TemporaryDirectory() as cache_dir:
                items = [_make_item(i, cache_dir) for i in range(num_items)]
                dataset = _make_dataset(items, cache_dir)
                args = _make_args(cache_dir)
                acc = _make_accelerator(num_processes, pid)

                encoded_this_proc = []

                def encode(batch, _pid=pid):
                    encoded_this_proc.extend([(item.item_key, _pid) for item in batch])

                cache_latents.encode_datasets([dataset], encode, args, accelerator=acc)
                all_encoded.extend(encoded_this_proc)
        return all_encoded

    def test_no_item_missed(self):
        for n in [2, 3, 4]:
            num_items = 10
            all_encoded = self._simulate_all_processes(n, num_items)
            keys = [e[0] for e in all_encoded]
            expected = {f"item_{i:04d}" for i in range(num_items)}
            self.assertEqual(set(keys), expected, f"All items must be encoded for num_processes={n}")

    def test_no_item_duplicated(self):
        for n in [2, 3, 4]:
            num_items = 10
            all_encoded = self._simulate_all_processes(n, num_items)
            keys = [e[0] for e in all_encoded]
            self.assertEqual(len(keys), len(set(keys)), f"Each item encoded exactly once for num_processes={n}")

    def test_sharding_is_balanced(self):
        """Every process should receive at least one item; total must equal num_items.
        Hash-based sharding does not guarantee exact equality, but no process
        should be starved on any reasonably sized dataset."""
        n = 4
        num_items = 12
        per_process = {pid: [] for pid in range(n)}
        for pid in range(n):
            with tempfile.TemporaryDirectory() as cache_dir:
                items = [_make_item(i, cache_dir) for i in range(num_items)]
                dataset = _make_dataset(items, cache_dir)
                args = _make_args(cache_dir)
                acc = _make_accelerator(n, pid)

                def encode(batch, _pid=pid):
                    per_process[_pid].extend(batch)

                cache_latents.encode_datasets([dataset], encode, args, accelerator=acc)

        counts = [len(per_process[p]) for p in range(n)]
        self.assertEqual(sum(counts), num_items, "Total encoded items must equal dataset size")
        for p, c in enumerate(counts):
            self.assertGreater(c, 0, f"Process {p} must receive at least one item")

    def test_barrier_called_before_cleanup(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            items = [_make_item(i, cache_dir) for i in range(4)]
            dataset = _make_dataset(items, cache_dir)
            # Plant a stale file that the main process should delete
            stale_path = os.path.join(cache_dir, "stale_0000_0320x0320_hv.safetensors")
            open(stale_path, "w").close()
            dataset.get_all_latent_cache_files = lambda: [stale_path]

            args = _make_args(cache_dir)
            acc = _make_accelerator(2, 0)

            call_order = []
            original_wait = acc.wait_for_everyone.side_effect

            def track_wait():
                call_order.append("barrier")

            acc.wait_for_everyone.side_effect = track_wait

            original_remove = os.remove

            def track_remove(path):
                call_order.append(f"remove:{os.path.basename(path)}")

            encode = lambda batch: None

            with patch("os.remove", side_effect=track_remove):
                cache_latents.encode_datasets([dataset], encode, args, accelerator=acc)

            self.assertIn("barrier", call_order, "barrier must be called")
            barrier_idx = call_order.index("barrier")
            remove_entries = [i for i, e in enumerate(call_order) if e.startswith("remove:")]
            for ri in remove_entries:
                self.assertGreater(ri, barrier_idx, "remove must come after barrier")

    def test_cleanup_only_on_main_process(self):
        """Non-main processes must not call os.remove."""
        for pid in range(1, 4):
            with tempfile.TemporaryDirectory() as cache_dir:
                items = [_make_item(i, cache_dir) for i in range(4)]
                dataset = _make_dataset(items, cache_dir)
                stale_path = os.path.join(cache_dir, "stale.safetensors")
                open(stale_path, "w").close()
                dataset.get_all_latent_cache_files = lambda: [stale_path]

                args = _make_args(cache_dir)
                acc = _make_accelerator(4, pid)
                encode = lambda batch: None
                acc.wait_for_everyone.side_effect = None

                with patch("os.remove") as mock_remove:
                    cache_latents.encode_datasets([dataset], encode, args, accelerator=acc)
                    mock_remove.assert_not_called()

    def test_tqdm_disabled_on_non_main(self):
        """tqdm should receive disable=True for non-main processes."""
        with tempfile.TemporaryDirectory() as cache_dir:
            items = [_make_item(i, cache_dir) for i in range(2)]
            dataset = _make_dataset(items, cache_dir)
            args = _make_args(cache_dir)
            acc = _make_accelerator(2, 1)  # non-main
            encode = lambda batch: None

            with patch("musubi_tuner.cache_latents.tqdm") as mock_tqdm:
                mock_tqdm.return_value = iter([("key", items)])
                cache_latents.encode_datasets([dataset], encode, args, accelerator=acc)
                mock_tqdm.assert_called_once()
                _, kwargs = mock_tqdm.call_args
                self.assertTrue(kwargs.get("disable"), "tqdm must be disabled for non-main process")


# Tests for cache_text_encoder_outputs.process_text_encoder_batches


class TestProcessTEBatchesSingleProcess(unittest.TestCase):
    def test_all_items_encoded(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            items = [_make_item(i, cache_dir) for i in range(6)]
            dataset = _make_dataset(items, cache_dir)
            args = _make_args(cache_dir)

            all_files = [{}, {}]  # placeholder
            all_paths = [set(), set()]

            encoded = []

            def encode(batch):
                encoded.extend(batch)

            cache_te.process_text_encoder_batches(
                args.num_workers, args.skip_existing, args.batch_size,
                [dataset], all_files, all_paths, encode,
                accelerator=None,
            )
            self.assertEqual(len(encoded), 6)


class TestProcessTEBatchesMultiGPU(unittest.TestCase):
    def _simulate(self, num_processes: int, num_items: int):
        all_encoded = []
        for pid in range(num_processes):
            with tempfile.TemporaryDirectory() as cache_dir:
                items = [_make_item(i, cache_dir) for i in range(num_items)]
                dataset = _make_dataset(items, cache_dir)
                args = _make_args(cache_dir)
                acc = _make_accelerator(num_processes, pid)

                all_files = [set()]
                all_paths = [set()]

                encoded_this = []

                def encode(batch, _pid=pid):
                    encoded_this.extend([(item.item_key, _pid) for item in batch])

                cache_te.process_text_encoder_batches(
                    args.num_workers, args.skip_existing, args.batch_size,
                    [dataset], all_files, all_paths, encode,
                    accelerator=acc,
                )
                all_encoded.extend(encoded_this)
        return all_encoded

    def test_no_item_missed(self):
        for n in [2, 3, 4]:
            all_encoded = self._simulate(n, num_items=9)
            keys = [e[0] for e in all_encoded]
            expected = {f"item_{i:04d}" for i in range(9)}
            self.assertEqual(set(keys), expected, f"All 9 items must be covered for N={n}")

    def test_no_item_duplicated(self):
        for n in [2, 3, 4]:
            all_encoded = self._simulate(n, num_items=9)
            keys = [e[0] for e in all_encoded]
            self.assertEqual(len(keys), len(set(keys)), f"No duplicates for N={n}")


# Tests for cache_text_encoder_outputs.post_process_cache_files


class TestPostProcessCacheFiles(unittest.TestCase):
    def test_barrier_then_cleanup_main_only(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            stale = os.path.join(cache_dir, "stale.safetensors")
            open(stale, "w").close()

            all_files = [{os.path.normpath(stale)}]
            all_paths = [set()]  # stale not in valid paths

            dataset_mock = MagicMock()
            acc = _make_accelerator(2, 0)

            call_order = []
            acc.wait_for_everyone.side_effect = lambda: call_order.append("barrier")

            with patch("os.remove", side_effect=lambda p: call_order.append(f"remove:{os.path.basename(p)}")):
                cache_te.post_process_cache_files([dataset_mock], all_files, all_paths, keep_cache=False, accelerator=acc)

            self.assertIn("barrier", call_order)
            barrier_idx = call_order.index("barrier")
            for i, e in enumerate(call_order):
                if e.startswith("remove:"):
                    self.assertGreater(i, barrier_idx)

    def test_non_main_does_not_delete(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            stale = os.path.join(cache_dir, "stale.safetensors")
            open(stale, "w").close()

            all_files = [{os.path.normpath(stale)}]
            all_paths = [set()]

            dataset_mock = MagicMock()
            acc = _make_accelerator(2, 1)  # non-main process

            with patch("os.remove") as mock_remove:
                cache_te.post_process_cache_files([dataset_mock], all_files, all_paths, keep_cache=False, accelerator=acc)
                mock_remove.assert_not_called()

    def test_no_accelerator_deletes_stale(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            stale = os.path.join(cache_dir, "stale.safetensors")
            open(stale, "w").close()

            all_files = [{os.path.normpath(stale)}]
            all_paths = [set()]

            dataset_mock = MagicMock()

            cache_te.post_process_cache_files([dataset_mock], all_files, all_paths, keep_cache=False, accelerator=None)
            self.assertFalse(os.path.exists(stale), "Stale file must be deleted in single-process mode")


# Tests for full path collection (cleanup correctness)


class TestFullPathCollection(unittest.TestCase):
    """All processes must collect the FULL set of valid paths, not just their shard."""

    def test_all_paths_collected_on_each_process(self):
        n = 4
        num_items = 8
        paths_per_process = {}
        for pid in range(n):
            with tempfile.TemporaryDirectory() as cache_dir:
                items = [_make_item(i, cache_dir) for i in range(num_items)]
                dataset = _make_dataset(items, cache_dir)
                args = _make_args(cache_dir)
                acc = _make_accelerator(n, pid)
                acc.wait_for_everyone.side_effect = None

                with patch("os.remove"):
                    all_files = [set()]
                    all_paths = [set()]

                    cache_te.process_text_encoder_batches(
                        args.num_workers, args.skip_existing, args.batch_size,
                        [dataset], all_files, all_paths, lambda batch: None,
                        accelerator=acc,
                    )
                    paths_per_process[pid] = set(all_paths[0])

        expected = {
            os.path.normpath(items[i].text_encoder_output_cache_path)  # type: ignore
            for i in range(num_items)
        }
        for pid in range(n):
            # We can't check exact paths since tmp dirs differ per process — just check count
            self.assertEqual(len(paths_per_process[pid]), num_items, f"Process {pid} must track all {num_items} paths")


if __name__ == "__main__":
    unittest.main(verbosity=2)
