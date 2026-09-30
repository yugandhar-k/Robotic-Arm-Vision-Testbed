# Data

Nothing here is committed except this file — the dataset is regenerated from the
simulator, so it is reproducible rather than stored.

## Regenerate

```bash
python run_sim.py --dataset 2000 --seed 42
```

Writes two paired sets, one per camera, sharing frame index and ground-truth class:

```
data/captures/eye_in_hand/      frame_NNNNN.png + labels.jsonl
data/captures/fixed_overhead/   frame_NNNNN.png + labels.jsonl
```

Both are 224 x 224 RGB. The classification benchmark (`src/train.py`) reads these two
directories directly — see the top-level README's
[Classification Benchmark](../README.md#classification-benchmark) section for the models
trained on them.

## How the labels are produced

Every record's `class_id` is the object the arm is currently inspecting, read directly
from the simulator's own state — not from a detector, and not from manual annotation.
This is what lets the fixed-overhead label stay correct even on frames where the object is
exactly what the arm is occluding.

**Eye-in-hand** record fields:

| Field | Meaning |
|---|---|
| `file` | Image filename |
| `class_id`, `class_name` | Ground-truth class |
| `colour_rgb` | The object's colour — sampled independently of class, so it is a distractor, not a feature |
| `object_xyz` | World position of the object at capture time |
| `image_size` | Frame resolution |
| `layout_seed` | Seed of the layout the frame came from |
| `hover_height_m` | Height of the grasp frame above the table for this view |
| `centre_offset_px` | Labelled object's centroid offset from the image centre, from the claw camera's own segmentation |
| `auto_label_agrees` | Whether that independent segmentation-based check agrees with the ground truth (a pipeline sanity check, not the training label) |

**Fixed-overhead** record fields: as above except `centre_offset_px` and
`auto_label_agrees` are replaced by `occluded_fraction` — the share of the (fixed-size,
class-independent) crop window that is *not* the target object, read from the overhead
camera's segmentation buffer over the same box the crop was cut from.

## Determinism

Object layout, class assignment, colours and viewpoint jitter are all driven by `--seed`.
The same seed reproduces the same dataset. The fixed-overhead frame for a given index is
captured from wherever the arm happened to be left over from the previous inspection
(see `run_sim.generate_dataset`'s docstring) — deterministic given the seed, but not
independent of capture order.

## Interactive single-frame capture

Pressing `C` in the viewer, or running with `--capture`, saves eye-in-hand frames into
`data/captures/eye_in_hand/` in the same format. Those frames are read back off the
on-screen claw-camera panel and downsampled to 224 x 224, so they are interchangeable
with headless ones. There is no interactive equivalent for the fixed-overhead view.
