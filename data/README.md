# Data

Nothing here is committed except this file — the dataset is regenerated from the
simulator, so it is reproducible rather than stored.

## Regenerate

```bash
python run_sim.py --dataset 500 --seed 42
```

Writes to `data/captures/`:

| File | Contents |
|---|---|
| `frame_NNNNN.png` | 224 x 224 RGB frame from the claw-mounted camera |
| `labels.jsonl` | One JSON record per frame |

Record fields:

| Field | Meaning |
|---|---|
| `file` | Image filename |
| `class_id`, `class_name` | Ground-truth class (`-1` / `background` if nothing is centred) |
| `colour_rgb` | The object's colour — sampled independently of class, so it is a distractor, not a feature |
| `object_xyz` | World position of the object at capture time |
| `image_size` | Frame resolution |
| `layout_seed` | Seed of the layout the frame came from |
| `hover_height_m` | Height of the grasp frame above the table for this view |
| `centre_offset_px` | Labelled object's centroid offset from the image centre |

## How the labels are produced

Labels come from MuJoCo's segmentation buffer, not from manual annotation. For each frame
the renderer produces a per-pixel geom-id map; whichever geom dominates the centre patch
(40 % of the image) is looked up in the scene's geom-id to class table. Frames where no
object covers at least 2 % of the centre patch are labelled `background` (`class_id: -1`).

Ground truth is therefore exact and costs nothing to scale. `centre_offset_px` is recorded
alongside so the framing quality of the set can be audited directly — see the Limitations
section of the top-level README for the clutter caveat this does not solve.

## Interactive capture

Pressing `C` in the viewer, or running with `--capture`, saves frames into the same
directory and format. Those frames are read back off the on-screen claw-camera panel and
downsampled to 224 x 224, so they are interchangeable with headless ones.

## Determinism

Object layout, class assignment, colours and viewpoint jitter are all driven by `--seed`.
The same seed reproduces the same dataset.
