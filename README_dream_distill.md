# Dream Imagination Distillation (DID)

## The finding this is built on

FastWAM's attention mask has a consequence that is easy to miss.
`wan_video_dit.py::build_video_to_video_mask` in `first_frame_causal` mode does:

```python
video_mask = torch.ones((S, S), dtype=torch.bool)     # fully connected
video_mask[:first_frame_tokens, first_frame_tokens:] = False
```

Despite the name, **only the first frame is causal**: frame 0 cannot see later
frames, while frames 1..N stay fully bidirectional. Combined with the joint MoT
mask in `fastwam/model.py::_build_mot_attention_mask`:

| query \ key | video frame 0 | video frames 1..N | dream | action |
|---|---|---|---|---|
| video frame 0 | yes | no | no | no |
| video 1..N | yes | yes | no | no |
| dream | yes | **no** | own modality | no |
| action | yes | **no** | yes | yes |

**Nothing reads the video expert's future-frame tokens.** The video branch
computes an imagined future at every layer and it reaches the policy only
through the shared *weights*, never through the *activations*. That is the
structural reason this architecture does not need test-time future imagination:
it was never able to consume it.

DreamFastWAM was supposed to be the channel that carries future information to
the action expert, but its mask restricts it to the current frame too — so Dream
must hallucinate the future from a single frame while being forbidden to read
the future the video expert already computed.

DID adds that channel, at training time only.

## What it does

Per selected MoT layer, the Dream tokens for a given future offset are pooled
and projected, and trained to match the video expert's pooled tokens for the
corresponding latent frame, under a stop-gradient:

```
L_did = mean over (layer, offset) of [ 1 - cos( P_l(pool(dream_o)), sg(pool(video_f)) ) ]
```

The stop-gradient prevents the degenerate solution where the teacher collapses its
future representation into something trivially predictable (the BYOL / SimSiam
asymmetry).

> **Caveat — in the shipped ladder the `detach()` is currently a no-op.** All four
> rung configs set `freeze_video_expert: true`, so the video expert has
> `requires_grad_(False)` and there is no gradient path to the teacher regardless.
> The teacher here is a *frozen pretrained* video expert, which is the standard and
> arguably better distillation setup — but it means the collapse hazard described
> above cannot occur in any configuration that will actually be run, and the
> `detach()` is defensive only. It becomes load-bearing the moment anyone sets
> `freeze_video_expert: false`.

The offset↔frame correspondence is exact, not heuristic. The VAE's temporal
downsample factor is 4 and `dream_target.future_offsets` are raw-frame offsets,
so **latent frame `f` is exactly offset `4f`**. Offsets that do not divide
evenly are rejected rather than rounded.

**Inference is unchanged.** Dream still reads only the current frame, the
Video+Dream KV prefill is still valid, and test-time cost is identical to a
stock DreamFastWAM.

## Cost, stated plainly

DID needs `lambda_video > 0`, because at `lambda_video == 0` the video branch is
fed a **single** latent frame (`dream_fastwam/model.py:622-631`) and no future
tokens exist. So the whole ladder runs the video branch over 9 latent frames
(~882 tokens) instead of 1 (~98). With `freeze_video_expert: true` the teacher
is frozen, so no gradient flows through it and no optimizer state is allocated
for its 5 B parameters — but the forward cost is real.

This also means **the ladder is not comparable to the published Dream numbers**
in `ckpt_eval_report.md`, which were all run at `lambda_video: 0.0`. That is why
R0 exists as a matched control.

## Relation to prior work — read this before calling it a method

DID is a **recombination of standard components**, not a new mechanism. Anyone
writing this up should say so plainly:

| Component | Prior art |
|---|---|
| auxiliary loss at intermediate layers, dropped at inference | Deeply-Supervised Nets (2015) |
| hidden-state distillation with a learned projector to reconcile widths | FitNets hint layers (2015) |
| stop-gradient on the teacher to prevent collapse | BYOL (2020) / SimSiam (2021) |
| information available only at training time, inference unchanged | Learning Using Privileged Information (Vapnik & Vashist, 2009); Learning by Cheating (2019) |
| intermediate layer of a **diffusion transformer** + projection head + cosine to a fixed representation, removed at sampling | REPA / representation alignment for DiTs (2024) — this is the closest match to the actual code |

The loss in `model.py` is `1 - cos(projector(pool(dream)), stopgrad(pool(video)))`.
That is SimSiam's `D(p, stopgrad(z))` with a `Linear` instead of an MLP predictor.
Nothing in it is new.

What is *specific* here is only the **teacher choice**: an intermediate hidden state
of a video *diffusion* expert, for a particular future latent frame, taken in the
same forward pass, across an MoT expert boundary where the student is denied that
information by an attention mask rather than by input modality. That is one
substitution away from REPA, and it is arguably a **downgrade** versus supervising
against the actual future frame — DID distils an *unvalidated imagination* of the
future rather than the future itself.

> These citations are from memory and were **not** verified against a live search
> (the sandbox has no working web search). Check them before putting them in a paper.
> In particular, look for 2025–2026 VLA work on distilling privileged future access
> into a current-frame-only student — if that exists, DID is already published.



| Rung | Config | What differs |
|---|---|---|
| R0 | `dream_distill_libero_goal_r0` | stock `dream_fastwam` model — **structurally** has no distillation head, not `lambda_did: 0` |
| R1 | `dream_distill_libero_goal` | + DID, teacher = future frames |
| R3 | `dream_distill_libero_goal_teacher_current` | teacher = current frame |
| R4 | `dream_distill_libero_goal_oracle` | Dream *attends* to future video tokens (train + test) |

Everything else — data, batch size, steps, LR, `lambda_video`, freeze setting —
is held identical across all four.

Note the default is `layers: [29]`, i.e. **one** distillation head at the last MoT
layer. The multi-layer variant (`model.dream_distill.layers=[10,20,29]`) is an
untested knob, not a shipped claim.

### R3 is the rung that decides this

`ckpt_eval_report.md` already contains an unfavourable prior. The two runs with
`data.train.dream_target.future_offsets=[0]` — i.e. Dream supervised on the
**current** frame, with no future at all — scored 97.0 / 96.2 / 94.8 / 94.2, the
best of the whole Dream group, while the default `[4]` runs sit at 92–95.

So the existing evidence leans towards *"Dream helps through multi-modal
representation supervision, not through the future"*. **DID only counts as
working if R1 beats R3**, not merely R0. Run R3 first and be willing to accept
that it wins.

### R4 bounds the whole idea

R4 is not deployable: it changes the mask at train and test time, breaks the
prefill (the model raises if you call `infer_action`), and needs a full video
forward at inference. Its only job is to answer "how much could a *perfect* DID
buy?". **If R4 is level with R0, the premise is dead** — and reporting that is a
sharper version of the paper's own claim, not a failed experiment.

## Running it

No new training entrypoint is needed; the model config's `_target_` does the
work, so `scripts/train.py` is the launcher:

```bash
# R0 matched control
bash scripts/train_zero1.sh 8 task=dream_distill_libero_goal_r0

# R1 treatment
bash scripts/train_zero1.sh 8 task=dream_distill_libero_goal

# R3 control
bash scripts/train_zero1.sh 8 task=dream_distill_libero_goal_teacher_current

# R4 oracle
bash scripts/train_zero1.sh 8 task=dream_distill_libero_goal_oracle

# knobs
bash scripts/train_zero1.sh 8 task=dream_distill_libero_goal \
  model.dream_distill.layers=[10,20,29] model.dream_distill.lambda_did=0.5
```

Watch `loss_did` plus the per-layer `did_l<L>_f<F>` entries in the loss dict.

Evaluate every rung with 3 seeds and compare means — the spread between repeated
evals of one checkpoint in `ckpt_eval_report.md` is 1–2 points, so a single eval
cannot separate these rungs.

```bash
PYTHONPATH=src python -m pytest tests/test_dream_distill.py -q
```

## Rollback list

This experiment is entirely new files; nothing outside them references it
(verified with grep). To remove it completely:

```
rm -r src/fastwam/models/wan22/dream_distill/
rm configs/model/dream_fastwam_distill.yaml
rm configs/task/dream_distill_libero_goal.yaml
rm configs/task/dream_distill_libero_goal_r0.yaml
rm configs/task/dream_distill_libero_goal_teacher_current.yaml
rm configs/task/dream_distill_libero_goal_oracle.yaml
rm tests/test_dream_distill.py
rm README_dream_distill.md
```

`src/fastwam/runtime.py` is deliberately untouched: the configs reach the
factory through `_target_`, and the experiment reuses `create_dream_fastwam`
rather than duplicating its config validation, so R0 and R1 share one
construction path.
