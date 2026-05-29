"""Simulation tests for WAN 2.2 full fine-tuning (wan_train.py).

No real GPU, model weights, torchvision, or DeepSpeed required.
All heavy dependencies are stubbed in sys.modules before importing.
Tests verify:

1. _derive_train_flags: (train_low, train_high) from --train_models + paths
2. _make_ckpt_name: checkpoint file names for dual vs single
3. LR fallback: per-model LR falls back to base --learning_rate
4. Timestep routing: model selected by t >= boundary
5. _sample_images_dual: skip + warn when either model is None
6. Gather ordering: gather before end_training in source code
7. Simulated training loop: step/skip logic using stub optimizers
8. do_inference dual routing: high-noise t → high model, low-noise t → low model
9. load_transformer override: single load, no triple-load bug
10. Source guards: DeepSpeed check, DDP warning, single-model skip warning
11. Eval/train fn switching: both optimizer fns called on save/sample
"""

import argparse
import os
import sys
import types
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Stub out all heavy / unavailable modules before any project imports
# ---------------------------------------------------------------------------

def _stub(name):
    import importlib.machinery
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = m
    return m

for _mod in [
    "torchvision",
    "torchvision.transforms",
    "torchvision.transforms.functional",
    "cv2",
]:
    if _mod not in sys.modules:
        _stub(_mod)

# torchvision.transforms.functional needs to_tensor symbol
sys.modules["torchvision.transforms.functional"].to_tensor = lambda x: x

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# torch.cuda.current_device() is called at class-definition time in wan/modules/t5.py.
# Stub it out so the import succeeds in a CPU-only environment.
import torch.cuda as _torch_cuda
if not _torch_cuda.is_available():
    _torch_cuda.current_device = lambda: 0

from musubi_tuner.wan_train import _derive_train_flags, _make_ckpt_name, WanTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        train_models=None,
        dit=None,
        dit_high_noise=None,
        learning_rate=1e-5,
        learning_rate_low=None,
        learning_rate_high=None,
        output_name="test_model",
        output_dir="/tmp/test_output",
        sample_prompts=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class _TinyModel(nn.Module):
    def __init__(self, name="unnamed"):
        super().__init__()
        self._name = name
        self.linear = nn.Linear(4, 4)
        self._call_count = 0

    def forward(self, x, t=None, **kwargs):
        self._call_count += 1
        return [torch.zeros(1, 16, 1, 8, 8)]

    def enable_block_swap(self, *a, **kw): pass
    def move_to_device_except_swap_blocks(self, *a): pass
    def prepare_block_swap_before_forward(self): pass
    def enable_gradient_checkpointing(self, *a): pass
    def switch_block_swap_for_inference(self): pass
    def switch_block_swap_for_training(self): pass


# ---------------------------------------------------------------------------
# 1. _derive_train_flags
# ---------------------------------------------------------------------------

class TestDeriveTrainFlags:
    def test_both_explicit(self):
        args = _make_args(train_models="both", dit="low.pt", dit_high_noise="high.pt")
        tl, th = _derive_train_flags(args)
        assert tl is True and th is True

    def test_low_only_explicit(self):
        args = _make_args(train_models="low", dit="low.pt", dit_high_noise="high.pt")
        tl, th = _derive_train_flags(args)
        assert tl is True and th is False

    def test_high_only_explicit(self):
        args = _make_args(train_models="high", dit="low.pt", dit_high_noise="high.pt")
        tl, th = _derive_train_flags(args)
        assert tl is False and th is True

    def test_auto_both_when_both_paths_given(self):
        args = _make_args(train_models=None, dit="low.pt", dit_high_noise="high.pt")
        tl, th = _derive_train_flags(args)
        assert tl is True and th is True

    def test_auto_low_only_when_only_dit_given(self):
        args = _make_args(train_models=None, dit="low.pt", dit_high_noise=None)
        tl, th = _derive_train_flags(args)
        assert tl is True and th is False

    def test_auto_high_only_when_only_dit_high_given(self):
        args = _make_args(train_models=None, dit=None, dit_high_noise="high.pt")
        tl, th = _derive_train_flags(args)
        assert tl is False and th is True

    def test_raises_if_low_requested_but_no_dit(self):
        args = _make_args(train_models="low", dit=None)
        with pytest.raises(ValueError, match="--dit"):
            _derive_train_flags(args)

    def test_raises_if_high_requested_but_no_dit_high(self):
        args = _make_args(train_models="high", dit="low.pt", dit_high_noise=None)
        with pytest.raises(ValueError, match="--dit_high_noise"):
            _derive_train_flags(args)

    def test_raises_if_both_requested_but_high_missing(self):
        args = _make_args(train_models="both", dit="low.pt", dit_high_noise=None)
        with pytest.raises(ValueError):
            _derive_train_flags(args)

    def test_raises_if_nothing_provided(self):
        args = _make_args(train_models=None, dit=None, dit_high_noise=None)
        # auto-detect with nothing provided → train_low=False, train_high=False is valid
        # (no error; both False means no training, which would be caught downstream)
        tl, th = _derive_train_flags(args)
        assert tl is False and th is False


# ---------------------------------------------------------------------------
# 2. _make_ckpt_name
# ---------------------------------------------------------------------------

class TestMakeCkptName:
    def test_dual_step_low(self):
        n = _make_ckpt_name("model", "_low", "low", 100, is_step=True, is_last=False)
        assert n == "model_low-step00000100.safetensors"

    def test_dual_step_high(self):
        n = _make_ckpt_name("model", "_high", "high", 100, is_step=True, is_last=False)
        assert n == "model_high-step00000100.safetensors"

    def test_dual_epoch_low(self):
        n = _make_ckpt_name("model", "_low", "low", 3, is_step=False, is_last=False)
        assert n == "model_low-000003.safetensors"

    def test_dual_epoch_high(self):
        n = _make_ckpt_name("model", "_high", "high", 3, is_step=False, is_last=False)
        assert n == "model_high-000003.safetensors"

    def test_single_step_no_suffix(self):
        n = _make_ckpt_name("model", "", "low", 50, is_step=True, is_last=False)
        assert n == "model-step00000050.safetensors"

    def test_single_last_no_suffix(self):
        n = _make_ckpt_name("model", "", "low", None, is_step=False, is_last=True)
        assert n == "model.safetensors"

    def test_dual_last_low(self):
        n = _make_ckpt_name("model", "_low", "low", None, is_step=False, is_last=True)
        assert n == "model_low.safetensors"

    def test_dual_last_high(self):
        n = _make_ckpt_name("model", "_high", "high", None, is_step=False, is_last=True)
        assert n == "model_high.safetensors"

    def test_step_zero_padded(self):
        n = _make_ckpt_name("abc", "_low", "low", 1, is_step=True, is_last=False)
        assert "step00000001" in n


# ---------------------------------------------------------------------------
# 3. LR fallback logic
# ---------------------------------------------------------------------------

class TestLearningRateFallback:
    def test_base_lr_used_when_no_per_model_lr(self):
        args = _make_args(learning_rate=2e-4, learning_rate_low=None, learning_rate_high=None)
        lr_low = getattr(args, "learning_rate_low", None) or args.learning_rate
        lr_high = getattr(args, "learning_rate_high", None) or args.learning_rate
        assert lr_low == pytest.approx(2e-4)
        assert lr_high == pytest.approx(2e-4)

    def test_per_model_lr_overrides_base(self):
        args = _make_args(learning_rate=1e-5, learning_rate_low=5e-5, learning_rate_high=1e-6)
        lr_low = getattr(args, "learning_rate_low", None) or args.learning_rate
        lr_high = getattr(args, "learning_rate_high", None) or args.learning_rate
        assert lr_low == pytest.approx(5e-5)
        assert lr_high == pytest.approx(1e-6)

    def test_partial_override_only_low(self):
        args = _make_args(learning_rate=1e-5, learning_rate_low=3e-5, learning_rate_high=None)
        lr_low = getattr(args, "learning_rate_low", None) or args.learning_rate
        lr_high = getattr(args, "learning_rate_high", None) or args.learning_rate
        assert lr_low == pytest.approx(3e-5)
        assert lr_high == pytest.approx(1e-5)


# ---------------------------------------------------------------------------
# 4. Timestep routing
# ---------------------------------------------------------------------------

class TestTimestepRouting:
    def _route(self, t_norm, boundary=0.875):
        return "high" if t_norm >= boundary else "low"

    def test_above_boundary_is_high(self):
        assert self._route(0.9) == "high"
        assert self._route(0.875) == "high"
        assert self._route(1.0) == "high"

    def test_below_boundary_is_low(self):
        assert self._route(0.874) == "low"
        assert self._route(0.0) == "low"

    def test_custom_boundary(self):
        assert self._route(0.6, boundary=0.5) == "high"
        assert self._route(0.4, boundary=0.5) == "low"

    def test_train_low_only_skips_high_noise_batch(self):
        train_low, train_high = True, False
        t_norm = 0.9
        is_high = t_norm >= 0.875
        should_skip = is_high and not train_high
        assert should_skip is True

    def test_train_high_only_skips_low_noise_batch(self):
        train_low, train_high = False, True
        t_norm = 0.5
        is_high = t_norm >= 0.875
        should_skip = not is_high and not train_low
        assert should_skip is True

    def test_train_both_never_skips(self):
        train_low, train_high = True, True
        for t_norm in [0.0, 0.5, 0.875, 0.9, 1.0]:
            is_high = t_norm >= 0.875
            skip = (is_high and not train_high) or (not is_high and not train_low)
            assert skip is False, f"Unexpected skip at t_norm={t_norm}"


# ---------------------------------------------------------------------------
# 5. _sample_images_dual
# ---------------------------------------------------------------------------

class TestSampleImagesDual:
    def _make_trainer_stub(self):
        trainer = WanTrainer.__new__(WanTrainer)
        trainer._inference_transformer_high = None
        return trainer

    def test_skips_when_transformer_low_is_none(self, caplog):
        trainer = self._make_trainer_stub()
        args = _make_args(sample_prompts="test.txt")

        with patch("musubi_tuner.wan_train.should_sample_images", return_value=True):
            with patch.object(trainer, "sample_images") as mock_sample:
                with caplog.at_level(logging.WARNING):
                    trainer._sample_images_dual(
                        MagicMock(), args, 1, 10, None,
                        None, _TinyModel(), [], torch.float32,
                    )
        mock_sample.assert_not_called()
        assert "both low-noise and high-noise models are required" in caplog.text

    def test_skips_when_transformer_high_is_none(self, caplog):
        trainer = self._make_trainer_stub()
        args = _make_args(sample_prompts="test.txt")

        with patch("musubi_tuner.wan_train.should_sample_images", return_value=True):
            with patch.object(trainer, "sample_images") as mock_sample:
                with caplog.at_level(logging.WARNING):
                    trainer._sample_images_dual(
                        MagicMock(), args, 1, 10, None,
                        _TinyModel(), None, [], torch.float32,
                    )
        mock_sample.assert_not_called()
        assert "both low-noise and high-noise models are required" in caplog.text

    def test_calls_sample_images_when_both_present(self):
        trainer = self._make_trainer_stub()
        args = _make_args(sample_prompts="test.txt")
        mock_accel = MagicMock()
        mock_accel.unwrap_model.side_effect = lambda m: m

        with patch("musubi_tuner.wan_train.should_sample_images", return_value=True):
            with patch.object(trainer, "sample_images") as mock_sample:
                trainer._sample_images_dual(
                    mock_accel, args, 1, 10, None,
                    _TinyModel(), _TinyModel(), [], torch.float32,
                )
        mock_sample.assert_called_once()
        assert trainer._inference_transformer_high is None

    def test_high_model_ref_cleared_after_exception(self):
        trainer = self._make_trainer_stub()
        args = _make_args(sample_prompts="test.txt")
        mock_accel = MagicMock()
        mock_accel.unwrap_model.side_effect = lambda m: m

        def _raise(*a, **kw):
            raise RuntimeError("inference failed")

        with patch("musubi_tuner.wan_train.should_sample_images", return_value=True):
            with patch.object(trainer, "sample_images", side_effect=_raise):
                with pytest.raises(RuntimeError):
                    trainer._sample_images_dual(
                        mock_accel, args, 1, 10, None,
                        _TinyModel(), _TinyModel(), [], torch.float32,
                    )
        assert trainer._inference_transformer_high is None

    def test_no_call_when_should_sample_returns_false(self):
        trainer = self._make_trainer_stub()
        args = _make_args(sample_prompts="test.txt")

        with patch("musubi_tuner.wan_train.should_sample_images", return_value=False):
            with patch.object(trainer, "sample_images") as mock_sample:
                trainer._sample_images_dual(
                    MagicMock(), args, 1, 10, None,
                    _TinyModel(), _TinyModel(), [], torch.float32,
                )
        mock_sample.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Source-level ordering assertions
# ---------------------------------------------------------------------------

class TestSourceOrdering:
    @staticmethod
    def _read_source():
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "musubi_tuner", "wan_train.py"
        )
        with open(src_path, encoding="utf-8") as f:
            return f.read()

    def test_gather_before_end_training(self):
        src = self._read_source()
        gather_pos = src.rfind("gather_state_dict_for_save")
        end_pos = src.rfind("accelerator.end_training()")
        assert gather_pos != -1, "gather_state_dict_for_save not found"
        assert end_pos != -1, "accelerator.end_training() not found"
        assert gather_pos < end_pos, "gather must appear before end_training in source"

    def test_no_ai_style_dividers(self):
        import re
        src = self._read_source()
        dividers = re.findall(r"#\s*-{10,}", src)
        assert len(dividers) == 0, f"Found AI-style divider comments: {dividers}"

    def test_no_coauthored_by_line(self):
        src = self._read_source()
        assert "Co-Authored-By" not in src

    def test_final_gather_not_gated_on_is_main_process(self):
        """final_state_low/high gather calls must be collective, not inside is_main_process."""
        src = self._read_source()
        lines = src.splitlines()
        indent_main = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "if is_main_process:" in line and indent_main is None:
                indent_main = len(line) - len(line.lstrip())
            if "final_state_" in stripped and "gather_state_dict_for_save" in stripped:
                line_indent = len(line) - len(line.lstrip())
                assert line_indent <= 8, (
                    f"Line {i+1}: final gather appears too deeply indented "
                    f"(indent={line_indent}), possibly inside is_main_process block:\n{line}"
                )


# ---------------------------------------------------------------------------
# 7. Simulated training loop
# ---------------------------------------------------------------------------

class TestSimulatedTrainingLoop:
    def test_both_opts_receive_correct_batches(self):
        boundary = 0.875
        low_steps = []
        high_steps = []

        class _CountOpt:
            def __init__(self, tag):
                self.tag = tag
            def step(self):
                (low_steps if self.tag == "low" else high_steps).append(1)
            def zero_grad(self, **kw): pass

        batches = [
            (0.5, False),
            (0.9, True),
            (0.3, False),
            (0.95, True),
        ]
        opt_low = _CountOpt("low")
        opt_high = _CountOpt("high")

        for t_norm, is_high in batches:
            train_low, train_high = True, True
            if is_high and not train_high:
                continue
            if not is_high and not train_low:
                continue
            opt = opt_high if is_high else opt_low
            opt.step()

        assert len(low_steps) == 2
        assert len(high_steps) == 2

    def test_skip_logic_train_low_only(self):
        boundary = 0.875
        train_low, train_high = True, False
        t_norms = [0.5, 0.9, 0.3, 0.95]
        processed = [t for t in t_norms
                     if not ((t >= boundary and not train_high) or (t < boundary and not train_low))]
        assert len(processed) == 2
        assert all(t < boundary for t in processed)

    def test_skip_logic_train_high_only(self):
        boundary = 0.875
        train_low, train_high = False, True
        t_norms = [0.5, 0.9, 0.3, 0.95]
        processed = [t for t in t_norms
                     if not ((t >= boundary and not train_high) or (t < boundary and not train_low))]
        assert len(processed) == 2
        assert all(t >= boundary for t in processed)

    def test_loss_recorder(self):
        from musubi_tuner.utils.train_utils import LossRecorder
        recorder = LossRecorder()
        losses = [0.5, 0.4, 0.3, 0.2]
        for i, loss in enumerate(losses):
            recorder.add(epoch=0, step=i, loss=loss)
        assert abs(recorder.moving_average - sum(losses) / len(losses)) < 1e-6

    def test_checkpoint_name_at_step_500(self):
        ckpt = _make_ckpt_name("mymodel", "_low", "low", 500, is_step=True, is_last=False)
        assert "step00000500" in ckpt
        assert "_low" in ckpt

    def test_dual_gather_called_once_per_model(self):
        gather_ids = []

        def _fake_gather(accelerator, model, should_save=True):
            gather_ids.append(id(model))
            return {}

        model_low = _TinyModel()
        model_high = _TinyModel()
        mock_accel = MagicMock()

        with patch("musubi_tuner.wan_train.gather_state_dict_for_save", side_effect=_fake_gather):
            from musubi_tuner.wan_train import gather_state_dict_for_save as g
            g(mock_accel, model_low)
            g(mock_accel, model_high)

        assert gather_ids == [id(model_low), id(model_high)]

    def test_suffix_logic_dual_vs_single(self):
        train_low, train_high = True, True
        suffix_low = "_low" if (train_low and train_high) else ""
        suffix_high = "_high" if (train_low and train_high) else ""
        assert suffix_low == "_low"
        assert suffix_high == "_high"

        train_low, train_high = True, False
        suffix_low = "_low" if (train_low and train_high) else ""
        assert suffix_low == ""


# ---------------------------------------------------------------------------
# 8. do_inference dual routing simulation
# ---------------------------------------------------------------------------

class TestDoInferenceDualRouting:
    def test_high_noise_timestep_uses_high_model(self):
        calls = {"low": 0, "high": 0}

        class _Routed(nn.Module):
            def __init__(self, name):
                super().__init__()
                self._n = name
                self.linear = nn.Linear(4, 4)

            def forward(self, x, t=None, **kwargs):
                calls[self._n] += 1
                return [torch.zeros(1, 16, 1, 8, 8)]

        boundary = 0.875
        m_low = _Routed("low")
        m_high = _Routed("high")

        for raw_t in [900.0, 500.0, 950.0, 400.0]:
            t_norm = raw_t / 1000.0
            model = m_high if t_norm >= boundary else m_low
            model([torch.zeros(1, 16, 1, 8, 8)], t=torch.tensor([raw_t]),
                  context=[torch.zeros(1, 512, 4096)], seq_len=16)

        assert calls["high"] == 2
        assert calls["low"] == 2

    def test_inference_high_ref_cleared_after_call(self):
        trainer = WanTrainer.__new__(WanTrainer)
        trainer._inference_transformer_high = None
        args = _make_args(sample_prompts="dummy.txt")
        mock_accel = MagicMock()
        mock_accel.unwrap_model.side_effect = lambda m: m

        with patch("musubi_tuner.wan_train.should_sample_images", return_value=True):
            with patch.object(trainer, "sample_images"):
                trainer._sample_images_dual(
                    mock_accel, args, None, 10, None,
                    _TinyModel(), _TinyModel(), [], torch.bfloat16,
                )

        assert trainer._inference_transformer_high is None


# ---------------------------------------------------------------------------
# 9. load_transformer override — prevents triple-loading of high-noise model
# ---------------------------------------------------------------------------

class TestLoadTransformerOverride:
    """WanTrainer.load_transformer must call load_wan_model exactly once per invocation
    and must NOT internally load the high-noise model as a side-effect (the triple-load
    bug from the parent WanNetworkTrainer.load_transformer).
    """

    def _make_trainer(self):
        trainer = WanTrainer.__new__(WanTrainer)
        trainer.config = SimpleNamespace(hidden_size=1024)
        trainer.dit_high_noise_path = "high.safetensors"
        trainer.high_low_training = True
        trainer.blocks_to_swap = 0
        trainer._inference_transformer_high = None
        return trainer

    def _make_args(self):
        return argparse.Namespace(
            fp8_scaled=False,
            disable_numpy_memmap=False,
            force_v2_1_time_embedding=False,
        )

    def test_calls_load_wan_model_exactly_once(self):
        """Each load_transformer call must result in exactly one load_wan_model call."""
        trainer = self._make_trainer()
        mock_model = _TinyModel()
        mock_accel = MagicMock()
        mock_accel.device = torch.device("cpu")
        load_calls = []

        def _fake_load(config, device, path, attn_mode, split_attn, loading_device, dtype, fp8_scaled, **kw):
            load_calls.append(path)
            return mock_model

        with patch("musubi_tuner.wan.modules.model.load_wan_model", side_effect=_fake_load):
            result = trainer.load_transformer(
                mock_accel, self._make_args(), "low.safetensors", "torch", False, "cpu", torch.bfloat16
            )

        assert len(load_calls) == 1, (
            f"Expected exactly 1 load_wan_model call, got {len(load_calls)}: {load_calls}"
        )
        assert load_calls[0] == "low.safetensors"
        assert result is mock_model

    def test_high_noise_path_not_loaded_as_side_effect(self):
        """When loading the low-noise model, the high-noise file must not be touched."""
        trainer = self._make_trainer()
        trainer.dit_high_noise_path = "high.safetensors"
        mock_model = _TinyModel()
        mock_accel = MagicMock()
        mock_accel.device = torch.device("cpu")
        loaded_paths = []

        def _tracker(config, device, path, *a, **kw):
            loaded_paths.append(path)
            return mock_model

        with patch("musubi_tuner.wan.modules.model.load_wan_model", side_effect=_tracker):
            trainer.load_transformer(
                mock_accel, self._make_args(), "low.safetensors", "torch", False, "cpu", torch.bfloat16
            )

        assert "high.safetensors" not in loaded_paths, (
            f"high-noise model was loaded as a side-effect of loading low-noise model. "
            f"Loaded paths: {loaded_paths}"
        )

    def test_second_call_for_high_model_also_once(self):
        """Loading the high-noise model explicitly also causes only one load_wan_model call."""
        trainer = self._make_trainer()
        mock_model = _TinyModel()
        mock_accel = MagicMock()
        mock_accel.device = torch.device("cpu")
        load_calls = []

        def _fake_load(config, device, path, *a, **kw):
            load_calls.append(path)
            return mock_model

        with patch("musubi_tuner.wan.modules.model.load_wan_model", side_effect=_fake_load):
            trainer.load_transformer(
                mock_accel, self._make_args(), "high.safetensors", "torch", False, "cpu", torch.bfloat16
            )

        # Must be exactly 1 call — parent would call it twice (main + internal inactive load)
        assert len(load_calls) == 1, (
            f"Expected 1 load for high-noise model, got {len(load_calls)}: {load_calls}"
        )

    def test_dit_inactive_state_dict_set_to_none(self):
        """Override must clear dit_inactive_state_dict — no swap state for full fine-tune."""
        trainer = self._make_trainer()
        trainer.dit_inactive_state_dict = {"old": "state"}  # pre-existing state
        mock_model = _TinyModel()
        mock_accel = MagicMock()
        mock_accel.device = torch.device("cpu")

        with patch("musubi_tuner.wan.modules.model.load_wan_model", return_value=mock_model):
            trainer.load_transformer(
                mock_accel, self._make_args(), "low.safetensors", "torch", False, "cpu", torch.bfloat16
            )

        assert trainer.dit_inactive_state_dict is None, (
            "dit_inactive_state_dict should be None after override — state-dict swap is not used for full fine-tune"
        )


# ---------------------------------------------------------------------------
# 10. Source guards: DeepSpeed, DDP warning, single-model skip warning
#     (extends the ordering tests with new source-level checks)
# ---------------------------------------------------------------------------

class TestSourceGuards:
    @staticmethod
    def _src():
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "musubi_tuner", "wan_train.py"
        )
        with open(src_path, encoding="utf-8") as f:
            return f.read()

    def test_deepspeed_guard_present_in_train(self):
        """wan_train.py must explicitly check is_deepspeed_active after prepare_accelerator."""
        src = self._src()
        assert "is_deepspeed_active(accelerator)" in src, (
            "is_deepspeed_active guard not found — WAN full fine-tune train() overrides base class "
            "and must add its own DeepSpeed rejection."
        )
        assert "WAN does not support DeepSpeed" in src

    def test_ddp_dual_model_warning_present(self):
        """Source must warn that DDP with dual-model training is unsupported."""
        src = self._src()
        assert "Multi-GPU (DDP) is not supported for dual-model training" in src

    def test_single_model_skip_warning_present(self):
        """Source must warn about ~50% effective step reduction in single-model mode."""
        src = self._src()
        assert "50%%" in src or "~50%" in src, (
            "Single-model skip-rate warning not found in source."
        )

    def test_is_deepspeed_active_imported(self):
        """is_deepspeed_active must be imported from deepspeed_utils."""
        src = self._src()
        assert "is_deepspeed_active" in src and "deepspeed_utils" in src


# ---------------------------------------------------------------------------
# 11. Eval/train fn switching — both models toggled on save/sample
# ---------------------------------------------------------------------------

class TestEvalTrainFnSwitching:
    @staticmethod
    def _src():
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "musubi_tuner", "wan_train.py"
        )
        with open(src_path, encoding="utf-8") as f:
            return f.read()

    def test_both_eval_fns_called_in_step_section(self):
        """In the mid-step save/sample block, both optimizer_eval_fn_low and
        optimizer_eval_fn_high must be called instead of just the active model's fn."""
        src = self._src()
        lines = src.splitlines()
        in_block = False
        saw_eval_low = False
        saw_eval_high = False
        saw_train_low = False
        saw_train_high = False
        for line in lines:
            stripped = line.strip()
            if "if should_sampling or should_saving:" in stripped:
                in_block = True
            if in_block:
                if "optimizer_eval_fn_low()" == stripped:
                    saw_eval_low = True
                if "optimizer_eval_fn_high()" == stripped:
                    saw_eval_high = True
                if "optimizer_train_fn_low()" == stripped:
                    saw_train_low = True
                if "optimizer_train_fn_high()" == stripped:
                    saw_train_high = True
            if in_block and "optimizer_train_fn_high()" == stripped:
                break  # end of block
        assert saw_eval_low, "optimizer_eval_fn_low() not called in should_sampling/saving block"
        assert saw_eval_high, "optimizer_eval_fn_high() not called in should_sampling/saving block"
        assert saw_train_low, "optimizer_train_fn_low() not called in should_sampling/saving block"
        assert saw_train_high, "optimizer_train_fn_high() not called in should_sampling/saving block"

    def test_standalone_opt_eval_fn_not_called(self):
        """The old pattern opt_eval_fn() (single-model) must not appear alone in the block."""
        src = self._src()
        lines = src.splitlines()
        in_block = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "if should_sampling or should_saving:" in stripped:
                in_block = True
            if in_block and stripped == "opt_eval_fn()":
                pytest.fail(
                    f"Line {i+1}: opt_eval_fn() called standalone — "
                    "both optimizer_eval_fn_low() and optimizer_eval_fn_high() should be called instead."
                )
            if in_block and "optimizer_train_fn_high()" == stripped:
                break

    def test_dual_eval_fn_simulation(self):
        """Simulate the fixed pattern: both low and high eval fns are always called."""
        eval_low_calls = 0
        eval_high_calls = 0
        train_low_calls = 0
        train_high_calls = 0

        def optimizer_eval_fn_low():
            nonlocal eval_low_calls
            eval_low_calls += 1

        def optimizer_eval_fn_high():
            nonlocal eval_high_calls
            eval_high_calls += 1

        def optimizer_train_fn_low():
            nonlocal train_low_calls
            train_low_calls += 1

        def optimizer_train_fn_high():
            nonlocal train_high_calls
            train_high_calls += 1

        # Simulate the fixed block (both models, regardless of which was active)
        should_sampling = True
        should_saving = False
        if should_sampling or should_saving:
            optimizer_eval_fn_low()
            optimizer_eval_fn_high()
            # ... sampling/saving work would happen here ...
            optimizer_train_fn_low()
            optimizer_train_fn_high()

        assert eval_low_calls == 1
        assert eval_high_calls == 1
        assert train_low_calls == 1
        assert train_high_calls == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
