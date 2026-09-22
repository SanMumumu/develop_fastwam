"""Unit tests for Dream Imagination Distillation.

These deliberately avoid constructing a real DreamFastWAM (that needs the 5 B
Wan2.2 weights). Instead they exercise the pieces where a mistake would be
silent and would invalidate the ablation:

* the teacher really is stop-gradiented (otherwise the video expert learns to
  make itself easy to predict and the distillation is meaningless);
* capture-off ``DistillMoT`` is numerically identical to ``MoT``;
* the dream token -> (modality, future offset) index arithmetic is right;
* offsets that do not land on a latent frame are rejected rather than rounded;
* the R4 oracle really does change the mask, and refuses the prefill.

Run with::

    PYTHONPATH=src python -m pytest tests/test_dream_distill.py -q
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

# The package pulls in fastwam.utils, which imports imageio for video writing.
for _name in ("imageio", "imageio_ffmpeg"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except ImportError:  # pragma: no cover - depends on the environment
            _stub = types.ModuleType(_name)
            _stub.__getattr__ = lambda _k: None  # type: ignore[attr-defined]
            sys.modules[_name] = _stub

from fastwam.models.wan22.action_dit import ActionDiT  # noqa: E402
from fastwam.models.wan22.dream_distill.mot import DistillMoT  # noqa: E402
from fastwam.models.wan22.dream_distill.model import DreamDistillFastWAM  # noqa: E402
from fastwam.models.wan22.mot import MoT  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _tiny_expert(hidden: int = 32, layers: int = 3) -> ActionDiT:
    return ActionDiT(
        action_dim=4,
        hidden_dim=hidden,
        ffn_dim=hidden * 2,
        num_heads=2,
        attn_head_dim=8,
        num_layers=layers,
        text_dim=16,
        freq_dim=16,
        eps=1e-6,
    )


def _tiny_mot(cls=MoT, layers: int = 3):
    torch.manual_seed(0)
    video = _tiny_expert(hidden=32, layers=layers)
    torch.manual_seed(1)
    action = _tiny_expert(hidden=24, layers=layers)
    return cls({"video": video, "action": action}, mot_checkpoint_mixed_attn=False)


def _mot_inputs(mot, sv: int = 6, sa: int = 4, batch: int = 2):
    torch.manual_seed(7)
    inner = mot.num_heads * mot.attn_head_dim
    embeds = {
        "video": torch.randn(batch, sv, 32),
        "action": torch.randn(batch, sa, 24),
    }
    freqs = {
        "video": torch.polar(torch.ones(sv, 1, mot.attn_head_dim // 2),
                             torch.randn(sv, 1, mot.attn_head_dim // 2)),
        "action": torch.polar(torch.ones(sa, 1, mot.attn_head_dim // 2),
                              torch.randn(sa, 1, mot.attn_head_dim // 2)),
    }
    t_mod = {
        "video": torch.zeros(batch, 6, 32),
        "action": torch.zeros(batch, 6, 24),
    }
    mask = torch.ones(sv + sa, sv + sa, dtype=torch.bool)
    context = {"video": None, "action": None}
    del inner
    return embeds, mask, freqs, context, t_mod


# ---------------------------------------------------------------------------
# DistillMoT
# ---------------------------------------------------------------------------
def test_capture_off_is_identical_to_plain_mot():
    """A DistillMoT with capture disabled must not perturb the forward pass."""
    plain = _tiny_mot(MoT)
    distill = _tiny_mot(DistillMoT)
    distill.load_state_dict(plain.state_dict())
    plain.eval(); distill.eval()

    args = _mot_inputs(plain)
    with torch.no_grad():
        out_a = plain(embeds_all=args[0], attention_mask=args[1], freqs_all=args[2],
                      context_all=args[3], t_mod_all=args[4])
        out_b = distill(embeds_all=args[0], attention_mask=args[1], freqs_all=args[2],
                        context_all=args[3], t_mod_all=args[4])
    for key in out_a:
        assert torch.equal(out_a[key], out_b[key]), f"{key} diverged with capture off"


def test_capture_records_requested_layers_only():
    mot = _tiny_mot(DistillMoT)
    mot.configure_capture(layers=[0, 2], experts=["video"])
    mot.enable_capture(True)
    args = _mot_inputs(mot)
    with torch.no_grad():
        mot(embeds_all=args[0], attention_mask=args[1], freqs_all=args[2],
            context_all=args[3], t_mod_all=args[4])
    snaps = mot.pop_snapshots()
    assert set(snaps) == {("video", 0), ("video", 2)}
    assert snaps[("video", 0)].shape == (2, 6, 32)
    # pop must clear, or activations (and their graph) leak across steps
    assert mot.pop_snapshots() == {}


def test_capture_layer0_is_the_forward_input():
    """Snapshot L is the state ENTERING layer L, so layer 0 == the raw input."""
    mot = _tiny_mot(DistillMoT)
    mot.configure_capture(layers=[0], experts=["video"])
    mot.enable_capture(True)
    args = _mot_inputs(mot)
    with torch.no_grad():
        mot(embeds_all=args[0], attention_mask=args[1], freqs_all=args[2],
            context_all=args[3], t_mod_all=args[4])
    assert torch.equal(mot.pop_snapshots()[("video", 0)], args[0]["video"])


def test_configure_capture_rejects_bad_input():
    mot = _tiny_mot(DistillMoT)
    with pytest.raises(ValueError, match="out of range"):
        mot.configure_capture(layers=[99], experts=["video"])
    with pytest.raises(ValueError, match="unknown expert"):
        mot.configure_capture(layers=[0], experts=["dream"])


# ---------------------------------------------------------------------------
# loss maths, exercised on a bare instance
# ---------------------------------------------------------------------------
class _FakeDreamExpert:
    """Minimal stand-in with the token layout DID indexes into."""

    def __init__(self, modalities, n_tokens, num_offsets, offsets):
        self.modalities = modalities
        self.num_future_offsets = num_offsets
        self.future_offsets = offsets
        for m in modalities:
            setattr(self, f"n_{m}", n_tokens)

    def modality_slices(self):
        start, out = 0, {}
        for m in self.modalities:
            length = self.num_future_offsets * getattr(self, f"n_{m}")
            out[m] = slice(start, start + length)
            start += length
        return out


def _bare_model(*, modalities=("dyn", "depth"), n_tokens=3, offsets=(16, 32), teacher="future"):
    model = object.__new__(DreamDistillFastWAM)
    # object.__new__ skips nn.Module.__init__, so the submodule registries do
    # not exist yet; initialise them before assigning any child module.
    torch.nn.Module.__init__(model)
    model.dream_expert = _FakeDreamExpert(list(modalities), n_tokens, len(offsets), list(offsets))
    model.did_teacher = teacher
    model._did_tokens_per_frame = 2
    model._did_temporal_factor = 4
    model._did_latent_frames = [o // 4 for o in offsets]
    model.did_layers = [0]
    return model


def test_dream_offset_pooling_selects_the_right_tokens():
    """Dream queries are offset-major inside each modality block."""
    model = _bare_model(modalities=("dyn", "depth"), n_tokens=3, offsets=(16, 32))
    # 2 modalities x 2 offsets x 3 tokens = 12; give every token its index.
    tokens = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
    # offset 0 -> dyn[0:3] and depth[6:9]  => {0,1,2,6,7,8}, mean 4.0
    assert model._did_pool_dream_offset(tokens, 0).item() == pytest.approx(4.0)
    # offset 1 -> dyn[3:6] and depth[9:12] => {3,4,5,9,10,11}, mean 7.0
    assert model._did_pool_dream_offset(tokens, 1).item() == pytest.approx(7.0)


def test_video_frame_pooling_and_bounds():
    model = _bare_model()
    video = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)  # 3 frames x 2 tokens
    assert torch.allclose(model._did_pool_video_frame(video, 0), torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(model._did_pool_video_frame(video, 2), torch.tensor([[9.0, 10.0]]))
    with pytest.raises(ValueError, match="video sequence is only"):
        model._did_pool_video_frame(video, 5)


def test_teacher_is_detached_and_student_is_not():
    """The core requirement: gradient must flow to Dream, never to Video."""
    model = _bare_model(modalities=("dyn",), n_tokens=2, offsets=(4,))
    model.did_projectors = torch.nn.ModuleDict({"0": torch.nn.Linear(3, 3)})

    video = torch.randn(1, 4, 3, requires_grad=True)   # 2 frames x 2 tokens
    dream = torch.randn(1, 2, 3, requires_grad=True)
    loss, stats = model._compute_did_loss({("video", 0): video, ("dream", 0): dream})
    loss.backward()

    assert dream.grad is not None, "student (dream) must receive gradient"
    assert video.grad is None, "teacher (video) must be stop-gradiented"
    assert "did_l0_f1" in stats


def test_teacher_current_uses_frame_zero():
    model = _bare_model(modalities=("dyn",), n_tokens=2, offsets=(4,), teacher="current")
    model.did_projectors = torch.nn.ModuleDict({"0": torch.nn.Linear(3, 3)})
    video = torch.randn(1, 4, 3)
    dream = torch.randn(1, 2, 3)
    _, stats = model._compute_did_loss({("video", 0): video, ("dream", 0): dream})
    assert "did_l0_f0" in stats, "teacher='current' must distil from latent frame 0"


def test_offsets_must_land_on_a_latent_frame():
    """A non-multiple offset would silently distil the wrong frame; reject it."""
    model = object.__new__(DreamDistillFastWAM)
    model.vae = types.SimpleNamespace(temporal_downsample_factor=4)
    model.dream_expert = _FakeDreamExpert(["dyn"], 3, 1, [5])   # 5 % 4 != 0
    model.did_teacher = "future"
    model.loss_lambda_video = 1.0
    with pytest.raises(ValueError, match="not a multiple of the"):
        model._did_validate_offsets()


def test_future_teacher_requires_the_video_branch():
    """lambda_video == 0 feeds a single frame, so there is no future to distil."""
    model = object.__new__(DreamDistillFastWAM)
    model.vae = types.SimpleNamespace(temporal_downsample_factor=4)
    model.dream_expert = _FakeDreamExpert(["dyn"], 3, 1, [16])
    model.did_teacher = "future"
    model.loss_lambda_video = 0.0
    with pytest.raises(ValueError, match="requires lambda_video > 0"):
        model._did_validate_offsets()


def test_all_zero_offsets_rejected_for_future_teacher():
    model = object.__new__(DreamDistillFastWAM)
    model.vae = types.SimpleNamespace(temporal_downsample_factor=4)
    model.dream_expert = _FakeDreamExpert(["dyn"], 3, 1, [0])
    model.did_teacher = "future"
    model.loss_lambda_video = 1.0
    with pytest.raises(ValueError, match="R3 control"):
        model._did_validate_offsets()


# ---------------------------------------------------------------------------
# R4 oracle
# ---------------------------------------------------------------------------
class _FakeVideoExpert:
    """Reproduces `first_frame_causal`: frame 0 sees only itself."""

    def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame, device):
        mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        first = min(video_tokens_per_frame, video_seq_len)
        mask[:first, first:] = False
        return mask


def _mask_model(access: str):
    model = object.__new__(DreamDistillFastWAM)
    torch.nn.Module.__init__(model)
    model.video_expert = _FakeVideoExpert()
    model.dream_expert = _FakeDreamExpert(["dyn"], 2, 1, [4])
    model.did_dream_video_access = access
    return model


def _build(access: str):
    model = _mask_model(access)
    return model._build_mot_attention_mask(
        video_seq_len=6,          # 3 latent frames x 2 tokens
        dream_seq_len=2,
        action_seq_len=2,
        video_tokens_per_frame=2,
        device=torch.device("cpu"),
    )


def test_default_mask_keeps_dream_on_the_current_frame_only():
    mask = _build("current")
    dream = slice(6, 8)
    assert mask[dream, 0:2].all(), "dream must see the current frame"
    assert not mask[dream, 2:6].any(), "dream must NOT see future frames by default"


def test_oracle_mask_opens_dream_to_the_future():
    mask = _build("all")
    dream = slice(6, 8)
    assert mask[dream, 0:6].all(), "R4 oracle: dream must see every video token"
    # and it must not disturb anything else
    base = _build("current")
    assert torch.equal(mask[0:6, :], base[0:6, :]), "video rows changed"
    assert torch.equal(mask[8:, :], base[8:, :]), "action rows changed"


def test_oracle_refuses_the_prefill_path():
    """The prefill assumes Dream is independent of the video denoising state."""
    model = _mask_model("all")
    with pytest.raises(RuntimeError, match="prefill is no longer valid"):
        model._prefill_video_dream_cache()
