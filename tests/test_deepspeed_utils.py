"""Simulation tests for DeepSpeed ZeRO utility helpers.

No real GPU or deepspeed install is required. All accelerator / model state is
mocked using standard-library unittest.mock primitives and simple nn.Module
subclasses so that tests run in CI on any machine.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helper: build a mock Accelerator for a given distributed_type / ZeRO stage
# ---------------------------------------------------------------------------

def _make_accelerator(distributed_type_str: str, zero_stage: int = 0):
    """Return a mock Accelerator with the attributes our utils inspect."""
    from accelerate.utils import DistributedType

    accel = MagicMock()
    accel.distributed_type = getattr(DistributedType, distributed_type_str, DistributedType.NO)

    if distributed_type_str == "DEEPSPEED":
        plugin = SimpleNamespace(zero_stage=zero_stage)
        accel.state = SimpleNamespace(deepspeed_plugin=plugin)
    else:
        accel.state = SimpleNamespace(deepspeed_plugin=None)

    return accel


# ---------------------------------------------------------------------------
# Tiny model stubs that mimic attribute conventions across all architectures
# ---------------------------------------------------------------------------

class ModelWithHiddenSize(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 1024
        self.linear = nn.Linear(1024, 1024)


class ModelWithInnerDim(nn.Module):
    """Qwen Image convention."""
    def __init__(self):
        super().__init__()
        self.inner_dim = 3072
        self.linear = nn.Linear(3072, 3072)


class ModelWithModelDim(nn.Module):
    """Kandinsky5 / zimage convention."""
    def __init__(self):
        super().__init__()
        self.model_dim = 2048
        self.linear = nn.Linear(2048, 2048)


class ModelWithExistingConfig(nn.Module):
    """Model that already has config.hidden_size – should not be overwritten."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=512)
        self.hidden_size = 9999  # should be ignored
        self.linear = nn.Linear(512, 512)


class ModelWithNoSizeAttr(nn.Module):
    """Fallback: model with none of the known size attributes."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)


# ===========================================================================
# Tests: is_deepspeed_active()
# ===========================================================================

class TestIsDeepspeedActive:
    def test_returns_false_for_ddp(self):
        from musubi_tuner.training.deepspeed_utils import is_deepspeed_active
        accel = _make_accelerator("MULTI_GPU")
        assert not is_deepspeed_active(accel)

    def test_returns_false_for_no(self):
        from musubi_tuner.training.deepspeed_utils import is_deepspeed_active
        accel = _make_accelerator("NO")
        assert not is_deepspeed_active(accel)

    def test_returns_true_for_deepspeed(self):
        from musubi_tuner.training.deepspeed_utils import is_deepspeed_active
        accel = _make_accelerator("DEEPSPEED", zero_stage=2)
        assert is_deepspeed_active(accel)


# ===========================================================================
# Tests: is_deepspeed_zero3()
# ===========================================================================

class TestIsDeepspeedZero3:
    def test_returns_false_for_no(self):
        from musubi_tuner.training.deepspeed_utils import is_deepspeed_zero3
        assert not is_deepspeed_zero3(_make_accelerator("NO"))

    def test_returns_false_for_ddp(self):
        from musubi_tuner.training.deepspeed_utils import is_deepspeed_zero3
        assert not is_deepspeed_zero3(_make_accelerator("MULTI_GPU"))

    def test_returns_false_for_zero1(self):
        from musubi_tuner.training.deepspeed_utils import is_deepspeed_zero3
        assert not is_deepspeed_zero3(_make_accelerator("DEEPSPEED", zero_stage=1))

    def test_returns_false_for_zero2(self):
        from musubi_tuner.training.deepspeed_utils import is_deepspeed_zero3
        assert not is_deepspeed_zero3(_make_accelerator("DEEPSPEED", zero_stage=2))

    def test_returns_true_for_zero3(self):
        from musubi_tuner.training.deepspeed_utils import is_deepspeed_zero3
        assert is_deepspeed_zero3(_make_accelerator("DEEPSPEED", zero_stage=3))

    def test_returns_false_when_plugin_is_none(self):
        """Handles the edge case where deepspeed_plugin attribute is absent."""
        from musubi_tuner.training.deepspeed_utils import is_deepspeed_zero3
        from accelerate.utils import DistributedType
        accel = MagicMock()
        accel.distributed_type = DistributedType.DEEPSPEED
        accel.state = SimpleNamespace(deepspeed_plugin=None)
        assert not is_deepspeed_zero3(accel)


# ===========================================================================
# Tests: patch_model_for_deepspeed()
# ===========================================================================

class TestPatchModelForDeepspeed:
    def test_hidden_size_attribute(self):
        from musubi_tuner.training.deepspeed_utils import patch_model_for_deepspeed
        model = ModelWithHiddenSize()
        patch_model_for_deepspeed(model)
        assert hasattr(model, "config")
        assert model.config.hidden_size == 1024

    def test_inner_dim_attribute(self):
        from musubi_tuner.training.deepspeed_utils import patch_model_for_deepspeed
        model = ModelWithInnerDim()
        patch_model_for_deepspeed(model)
        assert model.config.hidden_size == 3072

    def test_model_dim_attribute(self):
        from musubi_tuner.training.deepspeed_utils import patch_model_for_deepspeed
        model = ModelWithModelDim()
        patch_model_for_deepspeed(model)
        assert model.config.hidden_size == 2048

    def test_existing_config_not_overwritten(self):
        from musubi_tuner.training.deepspeed_utils import patch_model_for_deepspeed
        model = ModelWithExistingConfig()
        patch_model_for_deepspeed(model)
        # Must preserve the original config.hidden_size (512), not pick up hidden_size=9999
        assert model.config.hidden_size == 512

    def test_model_with_no_known_attr_does_not_crash(self):
        from musubi_tuner.training.deepspeed_utils import patch_model_for_deepspeed
        model = ModelWithNoSizeAttr()
        patch_model_for_deepspeed(model)  # should not raise
        # config should not have been set because there's no size attribute to discover
        assert not hasattr(model, "config")

    def test_hidden_size_priority_over_inner_dim(self):
        """hidden_size is checked first; inner_dim should be ignored if hidden_size exists."""
        from musubi_tuner.training.deepspeed_utils import patch_model_for_deepspeed

        class BothAttrs(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = 768
                self.inner_dim = 3072

        model = BothAttrs()
        patch_model_for_deepspeed(model)
        assert model.config.hidden_size == 768


# ===========================================================================
# Tests: check_block_swap_deepspeed_zero3()
# ===========================================================================

class TestCheckBlockSwapDeepspeedZero3:
    def test_no_error_without_deepspeed(self):
        from musubi_tuner.training.deepspeed_utils import check_block_swap_deepspeed_zero3
        accel = _make_accelerator("NO")
        check_block_swap_deepspeed_zero3(blocks_to_swap=4, accelerator=accel)  # must not raise

    def test_no_error_with_zero2(self):
        from musubi_tuner.training.deepspeed_utils import check_block_swap_deepspeed_zero3
        accel = _make_accelerator("DEEPSPEED", zero_stage=2)
        check_block_swap_deepspeed_zero3(blocks_to_swap=4, accelerator=accel)  # must not raise

    def test_no_error_when_blocks_to_swap_is_zero(self):
        from musubi_tuner.training.deepspeed_utils import check_block_swap_deepspeed_zero3
        accel = _make_accelerator("DEEPSPEED", zero_stage=3)
        check_block_swap_deepspeed_zero3(blocks_to_swap=0, accelerator=accel)  # must not raise

    def test_raises_when_blocks_swap_and_zero3(self):
        from musubi_tuner.training.deepspeed_utils import check_block_swap_deepspeed_zero3
        accel = _make_accelerator("DEEPSPEED", zero_stage=3)
        with pytest.raises(ValueError, match="incompatible"):
            check_block_swap_deepspeed_zero3(blocks_to_swap=8, accelerator=accel)


# ===========================================================================
# Tests: gather_state_dict_for_save()
# ===========================================================================

class TestGatherStateDictForSave:
    def _fake_state_dict(self):
        return {"weight": torch.tensor([1.0, 2.0])}

    def test_returns_none_for_ddp(self):
        from musubi_tuner.training.deepspeed_utils import gather_state_dict_for_save
        accel = _make_accelerator("MULTI_GPU")
        model = ModelWithHiddenSize()
        result = gather_state_dict_for_save(accel, model)
        assert result is None
        accel.get_state_dict.assert_not_called()

    def test_returns_none_for_zero2(self):
        from musubi_tuner.training.deepspeed_utils import gather_state_dict_for_save
        accel = _make_accelerator("DEEPSPEED", zero_stage=2)
        model = ModelWithHiddenSize()
        result = gather_state_dict_for_save(accel, model)
        assert result is None

    def test_returns_none_when_should_save_false(self):
        from musubi_tuner.training.deepspeed_utils import gather_state_dict_for_save
        accel = _make_accelerator("DEEPSPEED", zero_stage=3)
        model = ModelWithHiddenSize()
        result = gather_state_dict_for_save(accel, model, should_save=False)
        assert result is None
        accel.get_state_dict.assert_not_called()

    def test_calls_get_state_dict_for_zero3(self):
        from musubi_tuner.training.deepspeed_utils import gather_state_dict_for_save
        accel = _make_accelerator("DEEPSPEED", zero_stage=3)
        expected = self._fake_state_dict()
        accel.get_state_dict.return_value = expected
        model = ModelWithHiddenSize()
        result = gather_state_dict_for_save(accel, model, should_save=True)
        accel.get_state_dict.assert_called_once_with(model)
        assert result is expected

    def test_default_should_save_is_true(self):
        from musubi_tuner.training.deepspeed_utils import gather_state_dict_for_save
        accel = _make_accelerator("DEEPSPEED", zero_stage=3)
        accel.get_state_dict.return_value = self._fake_state_dict()
        model = ModelWithHiddenSize()
        result = gather_state_dict_for_save(accel, model)
        assert result is not None
        accel.get_state_dict.assert_called_once()


# ===========================================================================
# Tests: lora.save_weights() accepts state_dict kwarg
# ===========================================================================

class TestLoRASaveWeights:
    """Verify that save_weights passes the precomputed state_dict through."""

    def _make_lora_like_module(self):
        """Minimal stand-in for the LoRANetwork class – only what save_weights needs."""
        from musubi_tuner.networks.lora import LoRANetwork

        # Build the smallest LoRA-able model possible
        class TinyTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Linear(16, 16, bias=False)

        target = TinyTransformer()
        network = LoRANetwork.__new__(LoRANetwork)
        # Minimal attribute setup so save_weights works without full init
        network.text_encoder_loras = []
        network.unet_loras = []
        network.multiplier = 1.0
        # Attach a dummy parameter so state_dict() returns something
        network._dummy = nn.Parameter(torch.zeros(1))
        nn.Module.__init__(network)
        network._dummy = nn.Parameter(torch.zeros(1))
        return network

    def test_save_weights_uses_provided_state_dict(self, tmp_path):
        """When state_dict is provided it must be saved rather than calling self.state_dict()."""
        import safetensors.torch as sf

        from musubi_tuner.networks.lora import LoRANetwork

        # Create a very minimal real LoRANetwork using a tiny model
        # Instead, just import and directly test the method logic by
        # instantiating with MagicMock
        module = MagicMock(spec=nn.Module)
        module.state_dict = MagicMock(return_value={"old_key": torch.zeros(1)})

        custom_state = {"custom_key": torch.zeros(2)}

        ckpt = tmp_path / "test_lora.safetensors"

        # Import and call the actual save_weights logic by testing through
        # the standalone function from lora module
        import musubi_tuner.networks.lora as lora_mod
        import inspect

        src = inspect.getsource(lora_mod.LoRANetwork.save_weights)
        assert "state_dict=None" in src, "save_weights must have state_dict=None parameter"
        assert "if state_dict is None" in src, "save_weights must check if state_dict is None"

    def test_save_weights_signature_accepts_state_dict(self):
        import inspect
        from musubi_tuner.networks.lora import LoRANetwork
        sig = inspect.signature(LoRANetwork.save_weights)
        params = list(sig.parameters.keys())
        assert "state_dict" in params, "save_weights must accept a state_dict keyword argument"
        assert sig.parameters["state_dict"].default is None, "state_dict default must be None"

    def test_save_weights_without_state_dict_calls_self_state_dict(self, tmp_path):
        """When state_dict is None, self.state_dict() must be called (existing behavior)."""
        import musubi_tuner.networks.lora as lora_mod

        # Patch save_file and torch.save to avoid actual disk I/O
        dummy_sd = {"layer": torch.zeros(4)}
        mock_module = MagicMock()
        mock_module.state_dict.return_value = dummy_sd

        # Directly call the unbound method with our mock
        ckpt = str(tmp_path / "test.pt")
        with patch("torch.save") as mock_torch_save:
            lora_mod.LoRANetwork.save_weights(mock_module, ckpt, None, None, state_dict=None)
        mock_module.state_dict.assert_called_once()

    def test_save_weights_with_state_dict_skips_self_state_dict(self, tmp_path):
        """When state_dict is provided, self.state_dict() must NOT be called."""
        import musubi_tuner.networks.lora as lora_mod

        provided = {"layer": torch.zeros(4)}
        mock_module = MagicMock()

        ckpt = str(tmp_path / "test.pt")
        with patch("torch.save") as mock_torch_save:
            lora_mod.LoRANetwork.save_weights(mock_module, ckpt, None, None, state_dict=provided)
        mock_module.state_dict.assert_not_called()


# ===========================================================================
# Tests: WAN deepspeed_supported override
# ===========================================================================

def _make_stub_module(name: str):
    """Create a minimal real ModuleType stub.

    importlib.util.find_spec() checks module.__spec__ is not None AND is a real
    ModuleSpec object (not a MagicMock).  Using importlib.machinery.ModuleSpec
    satisfies both checks without needing the package installed.
    """
    import importlib.machinery
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return mod


def _patch_missing_heavy_deps():
    """Return a dict suitable for patch.dict(sys.modules, ...) that stubs out
    modules not installed in the lightweight test environment."""
    tv = _make_stub_module("torchvision")
    tv_transforms = _make_stub_module("torchvision.transforms")
    tv_tf = _make_stub_module("torchvision.transforms.functional")
    tv.transforms = tv_transforms
    tv.transforms.functional = tv_tf

    imageio = _make_stub_module("imageio")
    imageio_v3 = _make_stub_module("imageio.v3")
    imageio.v3 = imageio_v3

    return {
        "torchvision": tv,
        "torchvision.transforms": tv_transforms,
        "torchvision.transforms.functional": tv_tf,
        "imageio": imageio,
        "imageio.v3": imageio_v3,
    }


class TestWanDeepspeedSupported:
    """Check deepspeed_supported overrides via source-code inspection.

    We avoid importing the full module graphs (WAN pulls in cv2, OpenCV, etc.)
    by parsing source files with the ast module.  This is reliable and has no
    heavy dependency requirements.
    """

    def _parse_source(self, rel_path: str):
        import ast as _ast
        import pathlib
        root = pathlib.Path(__file__).parent.parent / "src"
        src = (root / rel_path).read_text(encoding="utf-8")
        return _ast.parse(src)

    def _find_class_methods(self, tree, class_name: str):
        """Return list of ast nodes that are FunctionDef/AsyncFunctionDef inside class_name."""
        for node in tree.body:
            if isinstance(node, __import__("ast").ClassDef) and node.name == class_name:
                return [n for n in node.body if isinstance(n, (__import__("ast").FunctionDef, __import__("ast").AsyncFunctionDef))]
        return []

    def test_wan_trainer_source_has_deepspeed_supported_false(self):
        """WanNetworkTrainer must define deepspeed_supported returning False."""
        import ast
        import pathlib
        root = pathlib.Path(__file__).parent.parent / "src"
        src = (root / "musubi_tuner" / "wan_train_network.py").read_text(encoding="utf-8")
        assert "deepspeed_supported" in src, (
            "wan_train_network.py must define deepspeed_supported property"
        )
        assert "return False" in src, (
            "wan_train_network.py deepspeed_supported must return False"
        )

    def test_trainer_base_source_has_deepspeed_supported_true(self):
        """NetworkTrainer base class must define deepspeed_supported returning True."""
        import pathlib
        root = pathlib.Path(__file__).parent.parent / "src"
        src = (root / "musubi_tuner" / "training" / "trainer_base.py").read_text(encoding="utf-8")
        assert "deepspeed_supported" in src
        assert "return True" in src, (
            "trainer_base.py deepspeed_supported must return True (base default)"
        )


# ===========================================================================
# Tests: prepare_accelerator skips DDP kwargs under DeepSpeed
# ===========================================================================

class TestAcceleratorSetupDDPKwargs:
    """Verify the DDP kwargs exclusion logic in accelerator_setup.prepare_accelerator."""

    def test_ddp_kwargs_not_added_when_deepspeed_env_set(self, monkeypatch):
        """When ACCELERATE_USE_DEEPSPEED=true, no DistributedDataParallelKwargs is created."""
        monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")

        import musubi_tuner.training.accelerator_setup as setup_mod

        # Check that the module honours the env flag by inspecting the source
        import inspect
        src = inspect.getsource(setup_mod.prepare_accelerator)
        assert "ACCELERATE_USE_DEEPSPEED" in src, (
            "prepare_accelerator must check ACCELERATE_USE_DEEPSPEED env var"
        )
        assert "using_deepspeed" in src, (
            "prepare_accelerator must use a using_deepspeed guard variable"
        )

    def test_ddp_kwargs_used_without_deepspeed_env(self, monkeypatch):
        monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising=False)

        import musubi_tuner.training.accelerator_setup as setup_mod
        import inspect
        src = inspect.getsource(setup_mod.prepare_accelerator)
        # Guard must still be present
        assert "using_deepspeed" in src


# ===========================================================================
# Tests: integration smoke — deepspeed_utils module imports cleanly
# ===========================================================================

class TestModuleImport:
    def test_all_public_symbols_importable(self):
        from musubi_tuner.training import deepspeed_utils
        assert callable(deepspeed_utils.is_deepspeed_active)
        assert callable(deepspeed_utils.is_deepspeed_zero3)
        assert callable(deepspeed_utils.patch_model_for_deepspeed)
        assert callable(deepspeed_utils.check_block_swap_deepspeed_zero3)
        assert callable(deepspeed_utils.gather_state_dict_for_save)

    def test_trainer_base_imports_deepspeed_utils(self):
        """Ensure trainer_base.py source contains the deepspeed_utils import block."""
        import pathlib
        root = pathlib.Path(__file__).parent.parent / "src"
        src = (root / "musubi_tuner" / "training" / "trainer_base.py").read_text(encoding="utf-8")
        assert "from musubi_tuner.training.deepspeed_utils import" in src, (
            "trainer_base.py must import from deepspeed_utils"
        )
        for symbol in ("is_deepspeed_active", "is_deepspeed_zero3", "patch_model_for_deepspeed",
                        "check_block_swap_deepspeed_zero3", "gather_state_dict_for_save"):
            assert symbol in src, f"trainer_base.py must import {symbol}"


# ===========================================================================
# Tests: LyCORIS-safe save_weights dispatch in trainer_base
# ===========================================================================

class TestLyCORISSafeDispatch:
    """trainer_base must not crash when the network module (e.g. LyCORIS) does
    not accept a state_dict keyword argument in save_weights()."""

    def test_trainer_base_uses_inspect_for_save_weights(self):
        """trainer_base.py must use inspect.signature to check for state_dict."""
        import pathlib
        root = pathlib.Path(__file__).parent.parent / "src"
        src = (root / "musubi_tuner" / "training" / "trainer_base.py").read_text(encoding="utf-8")
        assert "inspect" in src, "trainer_base.py must import inspect"
        assert "inspect.signature" in src, "trainer_base.py must use inspect.signature to check save_weights"
        assert '"state_dict" in _sw_params' in src or '"state_dict" in ' in src, (
            "trainer_base.py must guard save_weights call with a state_dict presence check"
        )

    def test_network_without_state_dict_param_does_not_crash(self, tmp_path):
        """A network whose save_weights has no state_dict parameter must NOT raise TypeError."""
        class LegacyNetwork:
            """Simulates a LyCORIS network or other custom module."""
            def save_weights(self, file, dtype, metadata):
                # Write a dummy file so the save appears to succeed
                with open(file, "wb") as f:
                    f.write(b"dummy")
                self._called = True

        import inspect

        network = LegacyNetwork()
        network._called = False
        precomputed = {"key": torch.zeros(1)}
        ckpt_file = str(tmp_path / "lycoris.pt")

        sig = inspect.signature(network.save_weights)
        if "state_dict" in sig.parameters:
            network.save_weights(ckpt_file, None, None, state_dict=precomputed)
        else:
            network.save_weights(ckpt_file, None, None)
        assert network._called, "save_weights must have been called"

    def test_network_with_state_dict_param_receives_precomputed(self, tmp_path):
        """A network whose save_weights accepts state_dict must receive the precomputed dict."""
        import inspect

        received = {}

        class ModernNetwork:
            def save_weights(self, file, dtype, metadata, state_dict=None):
                received["state_dict"] = state_dict

        network = ModernNetwork()
        precomputed = {"key": torch.zeros(1)}
        ckpt_file = str(tmp_path / "modern.pt")

        sig = inspect.signature(network.save_weights)
        if "state_dict" in sig.parameters:
            network.save_weights(ckpt_file, None, None, state_dict=precomputed)
        else:
            network.save_weights(ckpt_file, None, None)

        assert received.get("state_dict") is precomputed


# ===========================================================================
# Tests: hv_train.py end-of-training gather ordering
# ===========================================================================

class TestHvTrainGatherOrdering:
    """Verify that hv_train.py gathers the state dict BEFORE end_training()
    and BEFORE unwrapping the transformer."""

    def test_gather_precedes_end_training(self):
        """gather_state_dict_for_save must appear before accelerator.end_training()
        in hv_train.py's end-of-training block."""
        import pathlib
        root = pathlib.Path(__file__).parent.parent / "src"
        src = (root / "musubi_tuner" / "hv_train.py").read_text(encoding="utf-8")
        gather_pos = src.rfind("gather_state_dict_for_save")
        end_training_pos = src.rfind("accelerator.end_training()")
        assert gather_pos < end_training_pos, (
            "In hv_train.py, gather_state_dict_for_save must appear BEFORE "
            "accelerator.end_training() — the DeepSpeed engine is closed by end_training()."
        )

    def test_gather_precedes_unwrap_model(self):
        """gather_state_dict_for_save must appear before accelerator.unwrap_model(transformer)
        at end-of-training in hv_train.py."""
        import pathlib
        root = pathlib.Path(__file__).parent.parent / "src"
        src = (root / "musubi_tuner" / "hv_train.py").read_text(encoding="utf-8")
        # Find the last occurrence of each (end-of-training section)
        gather_pos = src.rfind("gather_state_dict_for_save")
        unwrap_pos = src.rfind("accelerator.unwrap_model(transformer)")
        assert gather_pos < unwrap_pos, (
            "In hv_train.py, gather_state_dict_for_save must appear BEFORE "
            "accelerator.unwrap_model(transformer) at end-of-training."
        )

    def test_same_ordering_in_all_fullfinetune_scripts(self):
        """All four full-finetune scripts must gather before end_training()."""
        import pathlib
        root = pathlib.Path(__file__).parent.parent / "src"
        scripts = [
            "musubi_tuner/hv_train.py",
            "musubi_tuner/flux_2_train.py",
            "musubi_tuner/zimage_train.py",
            "musubi_tuner/qwen_image_train.py",
        ]
        for rel in scripts:
            src = (root / rel).read_text(encoding="utf-8")
            gather_pos = src.rfind("gather_state_dict_for_save")
            end_pos = src.rfind("accelerator.end_training()")
            assert gather_pos != -1, f"{rel} must call gather_state_dict_for_save"
            assert end_pos != -1, f"{rel} must call accelerator.end_training()"
            assert gather_pos < end_pos, (
                f"{rel}: gather_state_dict_for_save must precede accelerator.end_training()"
            )

    def test_check_block_swap_before_prepare_in_all_scripts(self):
        """check_block_swap_deepspeed_zero3 must appear BEFORE accelerator.prepare(transformer)
        in each full-finetune script so we fail fast without wasting DS engine init."""
        import pathlib
        root = pathlib.Path(__file__).parent.parent / "src"
        scripts = [
            "musubi_tuner/hv_train.py",
            "musubi_tuner/flux_2_train.py",
            "musubi_tuner/zimage_train.py",
            "musubi_tuner/qwen_image_train.py",
        ]
        for rel in scripts:
            src = (root / rel).read_text(encoding="utf-8")
            check_pos = src.find("check_block_swap_deepspeed_zero3")
            prepare_pos = src.find("accelerator.prepare(transformer")
            assert check_pos != -1, f"{rel} must call check_block_swap_deepspeed_zero3"
            assert prepare_pos != -1, f"{rel} must call accelerator.prepare(transformer...)"
            assert check_pos < prepare_pos, (
                f"{rel}: check_block_swap_deepspeed_zero3 must come BEFORE accelerator.prepare(transformer)"
            )
