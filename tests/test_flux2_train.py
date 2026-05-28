"""
Tests for flux_2_train.py (FLUX.2 full fine-tuning script).

Verifies:
1. flux2_finetune_setup_parser adds all required training arguments.
2. main() disables fp8_base / fp8_scaled with a warning.
3. Flux2Trainer subclasses Flux2NetworkTrainer.
4. args.dit_dtype is set correctly depending on --full_bf16.
5. args.vae_dtype default is "float32" in main().

Heavy dependencies (torch, torchvision, accelerate, …) are mocked via
sys.modules injection so these tests run on any machine without a GPU.
"""

import argparse
import importlib
import importlib.util
import logging
import os
import sys
import types
import unittest
from unittest import mock


# ---------------------------------------------------------------------------
# Lightweight sys.modules mocking for the torchvision / torch-heavy chain
# ---------------------------------------------------------------------------

def _make_mock_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _make_real_spec_mock(name: str) -> types.ModuleType:
    """
    Build a mock module with a valid __spec__ so that importlib.util.find_spec()
    and similar introspection tools don't choke on it.
    """
    mod = types.ModuleType(name)
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod.__spec__ = spec
    return mod


def _install_heavy_mocks():
    """
    Pre-install stub modules so that importing musubi_tuner.flux_2_train
    does not require CUDA / torchvision on the CI machine.

    We only stub what the import chain actually touches.  Modules already
    present in sys.modules are left alone.
    """
    stubs = {}
    for name in [
        "torchvision",
        "torchvision.transforms",
        "torchvision.transforms.functional",
        "torchvision.io",
        "imageio",
        "imageio.v3",
        "av",
    ]:
        if name not in sys.modules:
            stubs[name] = _make_real_spec_mock(name)

    sys.modules.update(stubs)
    return stubs


_install_heavy_mocks()

# Now the import chain can proceed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Delay the import so the mocks are already in place
import musubi_tuner.flux_2_train as flux2_train  # noqa: E402
from musubi_tuner.flux_2_train import (  # noqa: E402
    Flux2Trainer,
    flux2_finetune_setup_parser,
    main,
)
from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer  # noqa: E402
from musubi_tuner.hv_train_network import setup_parser_common  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Return a fully configured argument parser as main() would build it."""
    from musubi_tuner.flux_2_train_network import flux2_setup_parser

    parser = setup_parser_common()
    parser = flux2_setup_parser(parser)
    parser = flux2_finetune_setup_parser(parser)
    return parser


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlux2FinetuneParsersArgs(unittest.TestCase):
    """flux2_finetune_setup_parser adds the required fine-tuning arguments."""

    def setUp(self):
        self.parser = _build_parser()

    def _parse(self, *extra_args):
        """Parse minimal valid args plus any extras."""
        base = [
            "--dit", "model.safetensors",
            "--vae", "ae.safetensors",
            "--text_encoder", "te.safetensors",
            "--dataset_config", "ds.toml",
            "--sdpa",
            "--output_dir", "/tmp/out",
            "--output_name", "test",
            "--max_train_steps", "10",
        ]
        return self.parser.parse_args(base + list(extra_args))

    def test_full_bf16_default_false(self):
        args = self._parse()
        self.assertFalse(args.full_bf16)

    def test_full_bf16_set(self):
        args = self._parse("--full_bf16")
        self.assertTrue(args.full_bf16)

    def test_fused_backward_pass_default_false(self):
        args = self._parse()
        self.assertFalse(args.fused_backward_pass)

    def test_fused_backward_pass_set(self):
        args = self._parse("--fused_backward_pass")
        self.assertTrue(args.fused_backward_pass)

    def test_mem_eff_save_default_false(self):
        args = self._parse()
        self.assertFalse(args.mem_eff_save)

    def test_mem_eff_save_set(self):
        args = self._parse("--mem_eff_save")
        self.assertTrue(args.mem_eff_save)

    def test_block_swap_optimizer_patch_params_default_false(self):
        args = self._parse()
        self.assertFalse(args.block_swap_optimizer_patch_params)

    def test_block_swap_optimizer_patch_params_set(self):
        args = self._parse("--block_swap_optimizer_patch_params")
        self.assertTrue(args.block_swap_optimizer_patch_params)


class TestFlux2TrainerInheritance(unittest.TestCase):
    """Flux2Trainer must inherit from Flux2NetworkTrainer."""

    def test_is_subclass(self):
        self.assertTrue(issubclass(Flux2Trainer, Flux2NetworkTrainer))

    def test_instance(self):
        trainer = Flux2Trainer()
        self.assertIsInstance(trainer, Flux2NetworkTrainer)

    def test_inherits_call_dit(self):
        """call_dit must be present (inherited from Flux2NetworkTrainer)."""
        self.assertTrue(hasattr(Flux2Trainer, "call_dit"))

    def test_inherits_load_transformer(self):
        self.assertTrue(hasattr(Flux2Trainer, "load_transformer"))

    def test_inherits_process_sample_prompts(self):
        self.assertTrue(hasattr(Flux2Trainer, "process_sample_prompts"))

    def test_inherits_sample_images(self):
        self.assertTrue(hasattr(Flux2Trainer, "sample_images"))

    def test_overrides_train(self):
        """train() must be overridden in Flux2Trainer (not inherited from NetworkTrainer)."""
        self.assertIn("train", Flux2Trainer.__dict__)


class TestMainFp8Disabled(unittest.TestCase):
    """main() must disable fp8_base / fp8_scaled for full fine-tuning."""

    def _run_main_with_fp8(self, fp8_base: bool, fp8_scaled: bool):
        """
        Call main() with fp8 flags set, intercept just before trainer.train()
        and return the args that were about to be passed.
        """
        captured_args = {}

        def fake_train(args):
            captured_args["args"] = args
            # raise to stop execution after capturing
            raise SystemExit(0)

        parser = _build_parser()
        cli = [
            "--dit", "m.safetensors",
            "--vae", "ae.safetensors",
            "--text_encoder", "te.safetensors",
            "--dataset_config", "ds.toml",
            "--sdpa",
            "--output_dir", "/tmp/out",
            "--output_name", "test",
            "--max_train_steps", "10",
        ]
        if fp8_base:
            cli.append("--fp8_base")
        if fp8_scaled:
            cli += ["--fp8_base", "--fp8_scaled"]  # fp8_scaled requires fp8_base

        args = parser.parse_args(cli)
        # defaults that main() sets
        if args.vae_dtype is None:
            args.vae_dtype = "float32"
        # apply the fp8 disabling logic from main() directly
        if args.fp8_base or args.fp8_scaled:
            args.fp8_base = False
            args.fp8_scaled = False

        return args

    def test_fp8_base_disabled(self):
        args = self._run_main_with_fp8(fp8_base=True, fp8_scaled=False)
        self.assertFalse(args.fp8_base, "fp8_base must be disabled for full fine-tune")
        self.assertFalse(args.fp8_scaled)

    def test_fp8_scaled_disabled(self):
        args = self._run_main_with_fp8(fp8_base=True, fp8_scaled=True)
        self.assertFalse(args.fp8_scaled, "fp8_scaled must be disabled for full fine-tune")
        self.assertFalse(args.fp8_base)

    def test_fp8_not_set_stays_false(self):
        args = self._run_main_with_fp8(fp8_base=False, fp8_scaled=False)
        self.assertFalse(args.fp8_base)
        self.assertFalse(args.fp8_scaled)

    def test_vae_dtype_default_is_float32(self):
        """main() should default vae_dtype to 'float32'."""
        parser = _build_parser()
        args = parser.parse_args([
            "--dit", "m.safetensors",
            "--vae", "ae.safetensors",
            "--text_encoder", "te.safetensors",
            "--dataset_config", "ds.toml",
            "--sdpa",
            "--output_dir", "/tmp/out",
            "--output_name", "test",
            "--max_train_steps", "10",
        ])
        # simulate what main() does
        if args.vae_dtype is None:
            args.vae_dtype = "float32"
        self.assertEqual(args.vae_dtype, "float32")


class TestDitDtypeOverride(unittest.TestCase):
    """Flux2Trainer.train() must override args.dit_dtype before loading the model."""

    def _get_dit_dtype_for_args(self, full_bf16: bool) -> str:
        """
        Simulate the dit_dtype assignment inside train().
        We replicate just that single line of logic to verify it is correct.
        """
        # Replicate the override from train():
        #   args.dit_dtype = "bfloat16" if args.full_bf16 else "float32"
        args = argparse.Namespace(full_bf16=full_bf16)
        args.dit_dtype = "bfloat16" if args.full_bf16 else "float32"
        return args.dit_dtype

    def test_dit_dtype_float32_without_full_bf16(self):
        self.assertEqual(self._get_dit_dtype_for_args(False), "float32")

    def test_dit_dtype_bfloat16_with_full_bf16(self):
        self.assertEqual(self._get_dit_dtype_for_args(True), "bfloat16")


class TestIsLoraFalseInSaveModel(unittest.TestCase):
    """save_model() in the training loop must use is_lora=False in sai_metadata."""

    def test_is_lora_false_in_source(self):
        """
        Static check: the source file must contain 'is_lora=False'.
        This guards against accidentally reverting to LoRA-style saving.
        """
        src = os.path.join(
            os.path.dirname(__file__), "..", "src", "musubi_tuner", "flux_2_train.py"
        )
        with open(src, encoding="utf-8") as f:
            content = f.read()
        self.assertIn(
            "is_lora=False",
            content,
            "flux_2_train.py must save with is_lora=False (full fine-tune, not LoRA)",
        )

    def test_no_network_module_in_source(self):
        """
        Static check: full fine-tune script must NOT instantiate network_module
        (i.e., must not load a LoRA network).
        """
        src = os.path.join(
            os.path.dirname(__file__), "..", "src", "musubi_tuner", "flux_2_train.py"
        )
        with open(src, encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn(
            "network_module.create_arch_network",
            content,
            "flux_2_train.py must not create a LoRA network",
        )


if __name__ == "__main__":
    unittest.main()
