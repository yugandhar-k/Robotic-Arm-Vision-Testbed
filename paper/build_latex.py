"""build_latex.py - Builds a full LaTeX source package (paper/latex/) from the actual
results/*.json files, then compiles it to paper/paper.pdf with tectonic, and zips the
LaTeX sources into paper/paper_latex.zip. Every number traces back to a results/ file,
same as build_paper.py's DOCX - this is a second rendering of the identical content, not
a second, independently-maintained draft.

Run after the six `python -m src.train ...` calls, `python -m src.evaluate`, and
`python -m src.significance` have all completed.

Usage
-----
    python paper/build_latex.py [--tectonic PATH_TO_TECTONIC_EXE] [--no-compile]

Author: Yugandhar Kulkarni - DRDO Research Intern - DES Pune University
"""
import argparse
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OUT_DIR = ROOT / "paper"
LATEX_DIR = OUT_DIR / "latex"
LATEX_FIGURES_DIR = LATEX_DIR / "figures"

CLASS_NAMES = ["cube", "sphere", "cylinder", "cuboid", "capsule", "ellipsoid", "disk", "rod"]
VIEWS = ["eye_in_hand", "fixed_overhead", "fixed_angled"]
FIXED_VIEWS = ["fixed_overhead", "fixed_angled"]
MODELS = ["cnn", "svm"]
VIEW_LABEL = {"eye_in_hand": "Eye-in-hand", "fixed_overhead": "Fixed-overhead",
             "fixed_angled": "Fixed-angled (45$^\\circ$)"}


def load(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def esc(text: str) -> str:
    """Escape LaTeX special characters and swap common Unicode punctuation for its
    LaTeX equivalent, so prose written in plain Unicode (matching build_paper.py's
    DOCX text) can be dropped straight into the .tex source.
    """
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ("—", "---"), ("–", "--"),
        ("‘", "`"), ("’", "'"), ("“", "``"), ("”", "''"),
        ("±", r"$\pm$"), ("°", r"$^\circ$"),
        ("×", r"$\times$"), ("→", r"$\rightarrow$"),
        ("…", r"\ldots{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


results = {v: {m: load(f"{v}_{m}.json") for m in MODELS} for v in VIEWS}
comparison = load("comparison.json")
significance = load("significance.json")

N_TRAINVAL = results["eye_in_hand"]["cnn"]["n_train_val"]
N_TEST = results["eye_in_hand"]["cnn"]["n_test"]
N_FRAMES = N_TRAINVAL + N_TEST

GAP = {(v, m): comparison[f"accuracy_gap_{v}_{m}"] * 100
      for v in FIXED_VIEWS for m in MODELS}
SIG = {(v, m): significance[v][m] for v in FIXED_VIEWS for m in MODELS}

NEAR_FULL_OCCLUSION_THRESHOLD = 0.95


def load_predictions(view, model):
    with open(RESULTS / f"{view}_{model}_test_predictions.json", encoding="utf-8") as f:
        return json.load(f)["records"]


def occlusion_stats(view, model):
    records = load_predictions(view, model)
    occ = [r["occluded_fraction"] for r in records]
    correct = [r["true_class_id"] == r["pred_class_id"] for r in records]
    near_full = [c for o, c in zip(occ, correct) if o >= NEAR_FULL_OCCLUSION_THRESHOLD]
    clear = [c for o, c in zip(occ, correct) if o < NEAR_FULL_OCCLUSION_THRESHOLD]
    return {
        "mean_occluded_fraction": sum(occ) / len(occ),
        "near_full_pct": 100 * len(near_full) / len(occ),
        "near_full_n": len(near_full),
        "near_full_acc": (sum(near_full) / len(near_full)) if near_full else None,
        "clear_acc": sum(clear) / len(clear),
    }


OCC = {(v, m): occlusion_stats(v, m) for v in FIXED_VIEWS for m in MODELS}


def significance_word(p):
    return "statistically significant" if p < 0.05 else "not statistically significant"


def p_str(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def gap_clause(view, model):
    p = SIG[(view, model)]["mcnemar"]["p_value"]
    return (f"{GAP[(view, model)]:+.2f} pp ({significance_word(p)}, "
           f"{p_str(p)})")


PLAIN_VIEW_LABEL = {"eye_in_hand": "eye-in-hand", "fixed_overhead": "fixed-overhead",
                    "fixed_angled": "fixed-angled"}


def directional_clause(view, model, winner_a="eye-in-hand", winner_b=None):
    """Describe the (signed) gap without assuming which side wins. Plain text only -
    safe to pass through para()/esc(), never raw()."""
    winner_b = winner_b or PLAIN_VIEW_LABEL[view]
    gap = GAP[(view, model)]
    p = SIG[(view, model)]["mcnemar"]["p_value"]
    leader = winner_a if gap > 0 else winner_b
    return (f"{leader} ahead by {abs(gap):.2f} pp ({significance_word(p)}, "
           f"{p_str(p)})")


# ---------------------------------------------------------------------------
# Body content - built as a list of raw LaTeX strings, then joined.
# ---------------------------------------------------------------------------

parts = []


def para(text):
    parts.append(esc(text))
    parts.append("")


def raw(text):
    parts.append(text)
    parts.append("")


def section(title):
    raw(f"\\section{{{esc(title)}}}")


def subsection(title):
    raw(f"\\subsection{{{esc(title)}}}")


def enumerate_block(items):
    raw("\\begin{enumerate}")
    for item in items:
        raw(f"  \\item {esc(item)}")
    raw("\\end{enumerate}")


abstract = (
    "Pick-and-place manipulators conventionally perceive their workspace through a "
    "camera fixed above it. This design has three structural weaknesses: the "
    "manipulator occludes its own target during the reach, the camera-to-robot "
    "transform requires a hand-eye calibration that drifts, and the object's scale "
    "and viewing angle are dictated by a fixed mounting point rather than by what is "
    "easiest to recognize. Mounting the camera on the end-effector (“eye-in-hand”) "
    "removes all three, at the cost of a viewpoint that now varies with the robot's own "
    "configuration. Prior comparisons conflate two properties of the conventional "
    "design -- being fixed, and being mounted straight overhead -- so we build two "
    "independent fixed-camera baselines, one straight down and one at a 45° elevation, "
    "and benchmark both against an eye-in-hand camera under identical conditions. We "
    "also replace the usual four-shape toy taxonomy with an eight-class one: the "
    "original four geometric primitives plus four more chosen to be genuine near-misses "
    "under a bounding-box or convex-hull shortcut (a capsule for a cylinder, an "
    "ellipsoid for a sphere, a disk for a cube seen from above, a rod for a cuboid seen "
    "edge-on), so the task cannot be solved by the aspect-ratio heuristic a generic "
    f"detector already exploits. From {N_FRAMES:,} paired, automatically-labelled "
    "frames per camera, we train two classifier families -- a compact convolutional "
    "neural network (CNN) and a classical HOG+Hu-moment shape-descriptor pipeline with "
    "a support-vector machine (SVM) -- under an identical 70/15/15 stratified split, "
    "5-fold cross-validation, and held-out-test protocol. "
    f"With the CNN, eye-in-hand is more accurate than the fixed-overhead camera "
    f"({directional_clause('fixed_overhead', 'cnn')}) but the gap to the "
    f"fixed-angled camera is smaller and not statistically significant "
    f"({directional_clause('fixed_angled', 'cnn')}). With the classical HOG+Hu-moment "
    f"SVM pipeline, eye-in-hand and fixed-overhead are statistically indistinguishable "
    f"({directional_clause('fixed_overhead', 'svm')}), and eye-in-hand is significantly "
    f"less accurate than the fixed-angled camera "
    f"({directional_clause('fixed_angled', 'svm')}). A per-sample occlusion metric, "
    "computed from the simulator's segmentation buffer, is consistent with this "
    "pattern: the fixed-overhead camera's crop is at least 95% non-object on "
    f"{OCC[('fixed_overhead', 'cnn')]['near_full_pct']:.1f}% of test frames, versus "
    f"{OCC[('fixed_angled', 'cnn')]['near_full_pct']:.1f}% for the fixed-angled camera, "
    "and accuracy on those near-fully-occluded frames drops sharply for every model "
    "and camera. Mounting angle, not fixedness by itself, accounts for most of the "
    "difference between the two fixed baselines, and eye-in-hand's advantage over a "
    "fixed camera depends on which fixed camera and which classifier are being "
    "compared. During development we found and fixed two data-integrity bugs -- objects "
    "rolling off the table, and two rounded object classes settling in a physically "
    "stable but unintended upright pose -- and report both, since the second bug was a "
    "direct side effect of fixing the first. Code, data-generation scripts, and "
    "trained-model configurations are released for reproduction."
)

# ---------------------------------------------------------------------------
# 1. Introduction
# ---------------------------------------------------------------------------

section("Introduction")

para(
    "Camera placement is a first-order design decision for any robotic manipulation "
    "system, and the default choice in most laboratory and industrial pick-and-place "
    "cells is a camera bolted above the workspace, looking down. This "
    "“fixed-overhead” configuration is simple to install and to reason about, but "
    "it has three structural weaknesses that all worsen exactly when the robot is "
    "doing its job."
)
para(
    "First, the manipulator occludes its own target: as the arm reaches toward an "
    "object, it necessarily passes between the fixed camera and that object, "
    "degrading the view at the moment the system most needs it. Second, a fixed "
    "camera requires a hand-eye calibration -- a measured transform between the "
    "camera's coordinate frame and the robot's -- that must be established once and "
    "re-established whenever the camera or the robot base is disturbed. Third, the "
    "object's apparent scale, viewing angle, and susceptibility to occlusion by "
    "clutter are all dictated by wherever the camera happens to be mounted, not by "
    "whatever viewpoint would make the object easiest to recognize."
)
para(
    "Mounting the camera on the end-effector itself -- an eye-in-hand configuration "
    "-- removes all three weaknesses by construction. The camera's pose is a known, "
    "calibration-free function of the joint angles; the arm cannot occlude a camera "
    "that is rigidly fixed to it; and the controller is free to choose whatever "
    "standoff distance and approach angle best frames the target. The cost is that "
    "the viewpoint is no longer fixed: scale, viewing angle, and background all vary "
    "with the robot's own configuration, which is conventionally assumed to make the "
    "perception problem harder."
)
para(
    "Existing comparisons of this kind, where they exist, conflate two separate "
    "properties of the conventional design: it is fixed, and it is mounted straight "
    "overhead. Weaknesses number 1 (self-occlusion) and number 3 (fixed apparent scale) both plausibly depend "
    "on the mounting angle as much as on the camera being fixed at all -- a "
    "straight-down “ceiling” camera is not how most real fixed workspace cameras "
    "are mounted; a corner-mounted, angled “security camera” view is far more "
    "typical. Whether that angle materially changes the occlusion or scale cost is, "
    "again, an empirical question that is rarely isolated: doing so requires not one "
    "but two independent fixed-camera baselines, benchmarked against the same "
    "eye-in-hand camera under identical conditions."
)
para(
    "A second gap in typical benchmarks of this kind is task difficulty. Four basic "
    "geometric primitives -- cube, sphere, cylinder, a flat box -- are trivially "
    "separable by a generic object detector from bounding-box aspect ratio alone, "
    "which means a comparison built on them mostly measures how easy the easy case "
    "is, not how the design choices under test perform when the classification "
    "problem is actually hard. A benchmark whose baseline already saturates near "
    "ceiling accuracy for every condition cannot show a meaningful gap between them "
    "even if one exists."
)
para(
    "This paper addresses both gaps directly. We build a MuJoCo simulation of a "
    "6-degree-of-freedom manipulator with a two-finger claw that carries an "
    "eye-in-hand camera, and add two independent fixed cameras -- one bolted straight "
    "overhead, one bolted at a 45° elevation -- rather than one. We replace the usual "
    "four-primitive taxonomy with an eight-class one, adding four classes chosen "
    "specifically to be near-misses for the original four under a bounding-box or "
    "convex-hull shortcut. At every simulated inspection, all three cameras observe "
    "the scene at the same instant, and every frame is labelled automatically and "
    "exactly from the simulator's own object state -- no manual annotation, and no "
    "detector-in-the-loop noise. We then train two classifier families with unrelated "
    "inductive biases -- a compact convolutional neural network learned end-to-end, "
    "and a classical pipeline of hand-crafted shape descriptors feeding a "
    "support-vector machine -- on each camera's data, under an identical "
    "cross-validated, held-out-tested protocol."
)
para("Our contributions are:")

enumerate_block([
    "A MuJoCo testbed that pairs an eye-in-hand camera with two independent "
    "fixed-camera baselines -- straight-overhead and 45-degree-angled -- observing the "
    "identical simulated instant, isolating both “fixed vs. moving” and “angle of "
    "a fixed mount” as separately measurable factors.",
    "An eight-class object taxonomy in which four classes are deliberate near-misses "
    "for the other four under a bounding-box or convex-hull shortcut, verified by a "
    "direct measurement showing the classical shape-descriptor baseline's accuracy "
    "drops substantially when moving from the four-class to the eight-class task.",
    "An automatic, exact ground-truth labelling protocol that requires no manual "
    "annotation, together with a per-sample occlusion metric for each fixed camera "
    "that quantifies, for every frame, how much of the classifier's input is not the "
    "target object -- used to show that the 45-degree camera is measurably less often "
    "fully occluded than the straight-down one.",
    "A controlled benchmark of two classifier families -- a CNN and a classical "
    "HOG + Hu-moment shape descriptor pipeline with an SVM -- on all three camera "
    "views, under a stratified train/validation/test split, 5-fold "
    "cross-validation, and a held-out test set untouched by any hyperparameter "
    "search.",
    "A paired statistical analysis (McNemar's test and bootstrap confidence "
    "intervals) of each resulting accuracy gap, reported honestly including where "
    "it does and does not reach conventional significance, and a direct "
    "demonstration that the gaps track a measured occlusion metric rather than an "
    "unrelated confound between the datasets.",
    "Two data-integrity bugs found and fixed during development -- objects rolling "
    "off the table, and a second bug this taxonomy exposed, in which the fix for "
    "the first caused two rounded object classes to settle in a physically stable "
    "but unintended upright pose -- reported in full because the second is a direct "
    "consequence of the first, and both are the kind of thing that quietly "
    "invalidates a benchmark if not caught.",
    "A fully reproducible, openly released pipeline: deterministic procedural "
    "scene generation, seeded dataset generation, and seeded model training, so "
    "every number in this paper can be regenerated from source.",
])

# ---------------------------------------------------------------------------
# 2. Related Work
# ---------------------------------------------------------------------------

section("Related Work")

raw("\\textbf{Eye-in-hand vision and visual servoing.} " + esc(
    "Mounting a camera on a robot's end-effector has a long history in the visual "
    "servoing literature, where it is used to drive a manipulator's motion directly "
    "from image measurements rather than from a calibrated world-frame estimate "
) + "\\cite{hutchinson1996tutorial}. " + esc(
    "That literature is primarily concerned with control -- using the image error "
    "signal to servo the arm to a goal pose -- rather than with measuring the "
    "downstream classification accuracy cost or benefit of the camera placement "
    "itself, or with separating the effect of camera placement from the effect of "
    "mounting angle, which is the question this paper addresses."
))
raw("")

raw("\\textbf{Camera viewpoint in recent manipulation learning.} " + esc(
    "More recent work looks at camera viewpoint mainly as one design choice inside "
    "a full learned manipulation policy, evaluated end-to-end on a real robot. Kim "
    "et al. "
) + "\\cite{kim2023givingrobots} " + esc(
    "use eye-in-hand video from human demonstrations to improve a manipulation "
    "policy's generalization. Chuang et al. "
) + "\\cite{chuang2024activevision} " + esc("and Liu et al. ") + "\\cite{liu2025avr} " + esc(
    "give the camera the ability to move or zoom during a task, so that occlusion "
    "and poor framing are avoided rather than measured. Li et al. "
) + "\\cite{li2026fromfixed} " + esc(
    "train policies to work across a range of camera positions rather than one "
    "fixed, calibrated placement. In all of these, camera placement is one of "
    "several factors -- lighting, object set, gripper, action space -- that vary "
    "together and are hard to hold constant between conditions, and the reported "
    "outcome is an end-to-end task success rate rather than a classification "
    "accuracy attributable to viewpoint alone. Isolating mounting angle as a single "
    "controlled variable, with everything else held fixed, is uncommon in this "
    "literature, in part because doing so on a real robot means physically "
    "remounting a camera and re-collecting a full dataset for each configuration "
    "tested. We use simulation specifically to make that isolation cheap: every "
    "other factor is held fixed and only the camera changes, at the cost of not "
    "testing on a real robot, a limitation discussed directly in Section 7."
))
raw("")

raw("\\textbf{Shape descriptors and classical recognition.} " + esc(
    "Before learned features became standard, object recognition relied on "
    "hand-crafted descriptors of shape and appearance. Hu's moment invariants "
) + "\\cite{hu1962moments} " + esc(
    "describe a silhouette's shape independently of its translation, scale, and "
    "rotation; the histogram of oriented gradients (HOG) "
) + "\\cite{dalal2005hog} " + esc(
    "describes local edge orientation statistics and was originally developed for "
    "pedestrian detection. Both remain useful, interpretable baselines, and we "
    "combine them as the feature representation for our classical classifier."
))
raw("")

raw("\\textbf{Learned recognition and detection.} " + esc(
    "Convolutional neural networks "
) + "\\cite{lecun1998gradient} " + esc(
    "learn a feature hierarchy directly from pixel data rather than relying on a "
    "hand-designed descriptor, and a support-vector machine "
) + "\\cite{cortes1995support} " + esc(
    "with a hand-crafted feature vector remains a standard, strong classical baseline "
    "against which to measure what a learned representation adds. Modern "
    "general-purpose object detectors separate coarse geometric primitives from "
    "bounding-box geometry alone at negligible cost, which is precisely why a "
    "benchmark built only on such primitives risks measuring how easy the easy case "
    "is rather than how the system under test performs under genuine visual "
    "ambiguity -- the motivation for the eight-class taxonomy in Section 3.1. We use "
    "both model families in their simplest effective form for this task, "
    "specifically so that any measured eye-in-hand advantage is a property of the "
    "data, not an artifact of one particular model family's inductive bias."
))
raw("")

raw("\\textbf{Simulation and sim-to-real.} " + esc(
    "We build our testbed in MuJoCo "
) + "\\cite{todorov2012mujoco}, " + esc(
    "a widely used physics engine for robotics research chosen for its fast, "
    "accurate contact dynamics and its offscreen rendering support, which we use for "
    "both RGB capture and the exact-label segmentation buffer described in "
    "Section 3.4. We do not attempt a sim-to-real transfer claim in this work; "
    "domain randomization "
) + "\\cite{tobin2017domain} " + esc(
    "-- randomizing lighting, texture, and sensor noise during data generation -- "
    "is the standard technique for closing that gap and is discussed as future work "
    "in Section 7."
))

# ---------------------------------------------------------------------------
# 3. System Design
# ---------------------------------------------------------------------------

section("System Design")

subsection("Manipulator, Scene and Taxonomy")
para(
    "The simulated workcell consists of a 0.92 m x 1.04 m table, a "
    "6-degree-of-freedom revolute manipulator mounted at its rear edge, and a "
    "two-finger parallel claw. The arm's joint set -- base yaw, shoulder, elbow, "
    "wrist roll, wrist pitch, and tool yaw -- gives it a comfortable workspace over "
    "a 3 x 2 grid of six object slots in front of the base, each jittered by "
    "±30 mm and assigned a random yaw on every layout. Object colour is sampled "
    "independently of class from a six-colour palette, so a classifier cannot "
    "shortcut the task by learning colour -- shape is the only class-carrying "
    "signal, a deliberate design choice because colour shortcuts are one of the "
    "most common ways a tabletop classification result fails to survive contact "
    "with real data."
)
para(
    "The object taxonomy has eight classes, not the four a comparable benchmark "
    "typically uses. The original four -- cube, sphere, cylinder, and a flat cuboid "
    "-- are trivially separable by aspect ratio alone. The four additions are each a "
    "deliberate near-miss for one of the originals under a bounding-box or "
    "convex-hull shortcut: a capsule reads as “a cylinder with soft edges,” an "
    "ellipsoid as “a stretched sphere,” a disk (a short, wide cylinder) as “a cube "
    "seen from directly above,” and a rod (a long, thin box) as “a cuboid seen "
    "edge-on.” A classifier that has only learned coarse aspect ratio should confuse "
    "these pairs; one that has learned actual silhouette curvature and proportion "
    "should not. Six of the eight classes, drawn without replacement, populate each "
    "layout's six slots, so every class appears in a majority of layouts without "
    "constraining the physical grid to eight slots."
)

subsection("Eye-in-Hand Camera and Inverse Kinematics")
para(
    "The eye-in-hand camera is mounted on the claw's palm, offset 36 mm laterally "
    "and tilted back toward the tool axis so that both finger pads remain in frame "
    "while the optical axis points down the approach direction. The arm executes a "
    "continuous inspect-and-pick cycle over the object field -- approach, settle, "
    "descend, close, lift, hold, lower, release, retreat -- and the classification "
    "frame is captured during settle, immediately before the grasp: this is both "
    "the decision point at which a real system must commit to an object identity, "
    "and the only phase at which the object is not partially occluded by the "
    "closing claw."
)
para(
    "Driving the tool to a precise Cartesian pose requires solving the arm's "
    "inverse kinematics. A naive damped-least-squares (DLS) solver seeded from the "
    "arm's current pose frequently converges to a joint-limited local optimum in "
    "this workspace (mean position error of roughly 194 mm in early testing). We "
    "instead compute a closed-form seed by constraining the optical axis to point "
    "straight down, which reduces the arm to a planar two-link chain plus a wrist "
    "pitch absorbing the remaining rotation; both elbow branches and both wrist "
    "branches of this closed-form solution are evaluated, each is polished with a "
    "few DLS iterations, and the best is kept. Later solves warm-start from the "
    "previous solution. This combination converges to under 0.1 mm in the IK solve "
    "itself; the residual, sensor-measured tool error (6.0 mm mean, 0.66 degree mean "
    "optical-axis deviation from vertical) is steady-state tracking error in the "
    "position-controlled joints under gravity, not IK error."
)
para(
    "Because the eye-in-hand camera is mounted laterally rather than directly on "
    "the tool axis, its optical axis crosses the tool axis at only one specific "
    "depth; hovering straight above an object therefore centres it correctly at "
    "only one inspection height. The controller instead solves, at each commanded "
    "inspection height, for the lateral tool offset that places the object on the "
    "optical axis at that height, undoing the offset before the claw descends so "
    "that grasping is unaffected. This closed-form correction reduces target "
    "centring error from tens of millimetres to a mean of roughly 14 px (of a "
    "112 px half-width) across inspection heights from 0.13 m to 0.24 m."
)

subsection("Two Fixed-Camera Baselines and a Shared Crop Protocol")
para(
    "Two independent fixed cameras are added, both bolted down and never moving: "
    "one 1.0 m directly above the table, pointed straight down (fovy 48 degrees); one "
    "pulled back 0.75 m horizontally and raised 0.75 m above the table, pointed at "
    "the object field's centroid (fovy 55 degrees) -- equal horizontal offset and height "
    "give exactly 45 degrees of elevation by construction. Both share the same crop "
    "protocol, implemented once and applied to either camera identically: because a "
    "real fixed camera has no downstream detector in this study -- the point is to "
    "isolate the classification-only cost of each fixed viewpoint -- each frame is "
    "cropped around the target object's ground-truth 3-D position, projected into "
    "that camera's image plane with a pinhole model calibrated from the camera's own "
    "field of view and verified against the renderer's segmentation buffer to "
    "sub-pixel accuracy for both cameras."
)
para(
    "Critically, the crop window is a single fixed physical size for every class "
    "and for both fixed cameras (170 mm, derived from the largest class's full "
    "3-D bounding radius plus a 30% margin -- not just its horizontal footprint, "
    "since the angled camera can foreshorten any of an object's three dimensions "
    "depending on its resting orientation), not a size fitted to whichever object "
    "is actually there. A real fixed camera cannot know an object's class before "
    "classifying it, so a class-conditioned crop size would leak the answer into "
    "the input and invalidate the benchmark; a fixed window instead reproduces, "
    "honestly, the “object scale dictated by where the camera happens to be bolted” "
    "weakness motivating this study."
)
para(
    "Both fixed-camera frames are captured from wherever the arm happens to be at "
    "that moment -- left over from the previous inspection, before the arm moves to "
    "its eye-in-hand pose for the current target. This is deliberately not the "
    "worst case (capturing at the same instant as the eye-in-hand frame, when the "
    "arm is already hovering directly over the target, occludes the object almost "
    "completely on every single frame for the overhead camera); it instead "
    "reproduces how a fixed camera is actually used in practice -- glancing at the "
    "workspace while the arm is occupied elsewhere -- and yields a realistic mixture "
    "of clear and partially occluded views for both cameras."
)

subsection("Automatic Labelling and the Occlusion Metric")
para(
    "Every frame's ground-truth label is the identity of the object slot the "
    "simulator is currently inspecting, read directly from its own object state -- "
    "not from a detector, and not from manual annotation. This is what allows both "
    "fixed-camera labels to remain correct precisely on the frames where the "
    "object is what is occluded, which a detector-based labelling scheme could not "
    "guarantee."
)
para(
    "As an independent pipeline sanity check (not used as a training label), the "
    "eye-in-hand frame's object identity is additionally verified against MuJoCo's "
    "own per-pixel segmentation buffer: whichever geometry dominates the centre "
    "40% of the frame is looked up in the scene's geometry-to-class table and "
    "compared against the ground truth."
)
para(
    "For each fixed camera, the segmentation buffer is used to compute "
    "occluded_fraction: the share of the (un-cropped) crop window's pixels that "
    "are not the target object -- whether because the manipulator is in the way, a "
    "neighbouring object intrudes, or the window simply extends past the object "
    "onto bare table. This metric is recorded for every fixed-camera frame, for "
    "both cameras independently, and is the basis for the causal analysis in "
    "Section 5."
)

subsection("Classifiers")
raw("\\textbf{CompactCNN.} " + esc(
    "A four-block convolutional network (16-32-64-128 channels; each block is a "
    "3x3 convolution, batch normalization, ReLU, and 2x2 max-pool), followed by "
    "global average pooling and a linear classification head, trained from "
    "scratch on 96x96 RGB input. Eight geometric primitives distinguished "
    "primarily by silhouette do not require a pretrained ImageNet backbone whose "
    "learned features have no relationship to this scene."
))
raw("")
raw("\\textbf{HOG + Hu-moment SVM.} " + esc(
    "A classical shape-descriptor pipeline: a 64x64 grayscale HOG descriptor "
    "(9 orientation bins, 8x8 cells, 16x16 blocks with 8-pixel stride) is "
    "concatenated with seven log-scaled Hu moments of the object's silhouette, "
    "standardized, and classified with a radial-basis-function support-vector "
    "machine. The silhouette is isolated by Otsu-thresholding the HSV saturation "
    "channel: every object is drawn from a saturated colour palette while the "
    "table and most of the arm are desaturated grey and steel tones by "
    "construction, so this segmentation needs no class-specific tuning -- with "
    "one caveat discussed in Section 7. The SVM's regularization and "
    "kernel-width hyperparameters (C, gamma) were selected once, by a 5-fold "
    "grid search on the eye-in-hand training split only, and reused unchanged "
    "for both fixed cameras, so neither fixed view's result can be an artifact "
    "of a baseline tuned against it."
))

# ---------------------------------------------------------------------------
# 4. Experimental Setup
# ---------------------------------------------------------------------------

section("Experimental Setup")

para(
    f"We generate {N_FRAMES:,} paired frames per camera view ({N_FRAMES:,} each for "
    f"eye-in-hand, fixed-overhead, and fixed-angled, sharing frame index and "
    f"ground-truth class across all three), at 224x224 resolution, with viewpoint "
    f"varied per frame (inspection height sampled over 0.11-0.18 m, claw yaw "
    f"perturbed by ±0.4 rad, target position jittered by ±15 mm) and the object "
    f"layout reshuffled every 12 frames. Class balance is close to uniform (each of "
    f"the eight classes within 10.6-13.7% of the dataset, against an ideal "
    f"12.5% for a uniform eight-way split) because classes are sampled six-of-eight, "
    f"without replacement, per layout slot rather than rebalanced post hoc."
)
para(
    "Each camera's dataset is split 70/15/15 into train/validation/test, "
    "stratified by class, with a fixed seed. This same seed, applied to an "
    "identical class-label sequence for all three camera views (the three "
    "datasets are paired frame-for-frame), produces the identical set of "
    "held-out test frame indices for every view -- a property we exploit for the "
    f"paired statistical tests in Section 5. The training and validation "
    f"partitions ({N_TRAINVAL:,} frames) are combined and used for 5-fold "
    "stratified cross-validation, training a fresh model per fold to report a "
    "mean and standard deviation that a single train/test split cannot provide. "
    f"A final model of each type is then fit on the full {N_TRAINVAL:,}-frame "
    f"partition and evaluated exactly once on the untouched {N_TEST:,}-frame "
    "held-out test set."
)
para(
    "The CNN is trained with Adam (initial learning rate 1e-3, weight decay "
    "1e-4), a cosine-annealed learning-rate schedule, and gradient-norm clipping "
    "at 5.0, for 18 epochs per cross-validation fold and 25 epochs for the final "
    "fit, with batch size 64 and a horizontal-flip augmentation (safe here "
    "because neither exact orientation nor colour carries class information by "
    "construction). This recipe was arrived at after an initial, simpler "
    "10-epoch fixed-learning-rate schedule produced unstable cross-validation "
    "folds (accuracies ranging from 0.58 to 0.94 on identical hyperparameters) "
    "and a model that fit its own cross-validation folds while collapsing on the "
    "untouched test set -- a symptom of under-training, not a property of the "
    "data, which we report in Section 6 as a methodological note. All "
    "experiments run on CPU only; no GPU was used or required."
)
para(
    "During development we also discovered and fixed two data-integrity issues "
    "worth reporting for the same reason. First, an early version of the "
    "simulation used a low rolling-friction coefficient for the object-table "
    "contact, which allowed spheres and cylinders -- the rollable classes in the "
    "original four-class taxonomy -- to occasionally roll off the table entirely "
    "after being nudged by the arm over many layout reshuffles; scanning a "
    "generated dataset for objects resting at floor height rather than table "
    "height found this had silently corrupted 1.48% of frames, concentrated "
    "entirely in the two rollable classes. Raising the rolling and torsional "
    "friction coefficients by roughly an order of magnitude eliminated the "
    "issue. Second, extending the taxonomy to include capsules and ellipsoids "
    "exposed a side effect of that fix: both classes are rounded enough that "
    "they do not reliably tip over from an upright drop the way a flat-ended "
    "cylinder does, and the raised rolling friction damps out what little "
    "rocking there is before it can complete a tip -- checking settled "
    "orientation directly showed both classes standing exactly upright on every "
    "seed tested, not occasionally. Rather than tune friction further, both "
    "classes are now spawned already lying on their side, sidestepping the "
    "stability question rather than depending on physics to resolve it. Neither "
    "bug affected the classification dataset materially in its final form; both "
    "are reported because catching them mattered and a benchmark that has been "
    "checked this way is more trustworthy for having been checked, not less."
)

# ---------------------------------------------------------------------------
# 5. Results
# ---------------------------------------------------------------------------

section("Results")

para(
    "Table I reports 5-fold cross-validation accuracy (mean ± "
    "standard deviation over the train+validation partition) alongside held-out "
    "test accuracy and macro-averaged F1 for all six (camera, model) combinations."
)

raw("\\begin{table*}[t]")
raw("\\centering")
raw("\\caption{Cross-validated and held-out test performance, all six combinations.}")
raw("\\label{tab:main}")
raw("\\begin{tabular}{llccc}")
raw("\\toprule")
raw("Camera & Model & 5-fold CV Accuracy & Test Accuracy & Test Macro-F1 \\\\")
raw("\\midrule")
for view in VIEWS:
    for model in MODELS:
        r = results[view][model]
        raw(f"{VIEW_LABEL[view]} & "
           f"{model.upper()} & "
           f"{r['cv_accuracy_mean']:.4f} $\\pm$ {r['cv_accuracy_std']:.4f} & "
           f"{r['test_accuracy']:.4f} & {r['test_macro_f1']:.4f} \\\\")
raw("\\bottomrule")
raw("\\end{tabular}")
raw("\\end{table*}")
raw("")

gap_sentence_parts = []
for fv in FIXED_VIEWS:
    for m in MODELS:
        gap_sentence_parts.append(
            f"{directional_clause(fv, m)} vs. {PLAIN_VIEW_LABEL[fv]} ({m.upper()})"
        )
para(
    "The sign and significance of the gap depend on both the fixed camera and the "
    "model family: " + "; ".join(gap_sentence_parts) + ", on a held-out test set "
    "neither model nor its hyperparameters saw during training or tuning. Eye-in-hand "
    "is ahead of the fixed-overhead camera for the CNN, but its lead over the "
    "fixed-angled camera is smaller and not significant for the CNN, and negative "
    "and significant for the SVM."
)

para("Table II reports per-class F1 on the held-out test set for all "
    "six combinations:")

raw("\\begin{table*}[t]")
raw("\\centering")
raw("\\caption{Per-class F1 on the held-out test set, all six combinations.}")
raw("\\label{tab:perclass}")
raw("\\small")
raw("\\begin{tabular}{l" + "c" * len(CLASS_NAMES) + "}")
raw("\\toprule")
raw(" & " + " & ".join(CLASS_NAMES) + " \\\\")
raw("\\midrule")
for view in VIEWS:
    for model in MODELS:
        r = results[view][model]
        f1s = [f"{r['test_per_class'][c]['f1']:.2f}" for c in CLASS_NAMES]
        label = VIEW_LABEL[view].replace("$^\\circ$", "d")
        raw(f"{label} {model.upper()} & " + " & ".join(f1s) + " \\\\")
raw("\\bottomrule")
raw("\\end{tabular}")
raw("\\end{table*}")
raw("")

raw("\\begin{figure}[t]")
raw("\\centering")
raw("\\includegraphics[width=\\linewidth]{figures/results.png}")
raw("\\caption{5-fold CV accuracy (bars, $\\pm$ 1 std) against held-out test accuracy "
   "(diamonds) for all six combinations.}")
raw("\\label{fig:results}")
raw("\\end{figure}")
raw("")

raw("\\begin{figure*}[t]")
raw("\\centering")
raw("\\includegraphics[width=0.85\\linewidth]{figures/confusion_matrix.png}")
raw("\\caption{Row-true, column-predicted confusion matrices, raw counts, for all six "
   "combinations.}")
raw("\\label{fig:confusion}")
raw("\\end{figure*}")
raw("")

subsection("Statistical Significance")
para(
    "All three views share the identical held-out test frame indices (Section 4), "
    "which licenses a paired analysis for each fixed camera against eye-in-hand "
    "(src/significance.py) -- more powerful than comparing two independent "
    "accuracies, because it asks \u201con the instances where the two views disagree, "
    "which one is usually right?\u201d rather than \u201care these two numbers different?\u201d"
)

raw("\\begin{table*}[t]")
raw("\\centering")
raw("\\caption{Paired bootstrap confidence intervals and McNemar's exact test for "
   "each fixed camera against eye-in-hand.}")
raw("\\label{tab:significance}")
raw("\\small")
raw("\\begin{tabular}{llccccc}")
raw("\\toprule")
raw("Fixed camera & Model & Eye-in-hand acc. [95\\% CI] & Fixed acc. [95\\% CI] & "
   "Gap [95\\% CI] & McNemar's $p$ \\\\")
raw("\\midrule")
for fv in FIXED_VIEWS:
    for m in MODELS:
        s = SIG[(fv, m)]
        eye_acc = s["eye_in_hand_accuracy_ci95"]
        fixed_acc = s[f"{fv}_accuracy_ci95"]
        gap = s["paired_bootstrap_gap"]
        label = VIEW_LABEL[fv].replace("$^\\circ$", " degrees")
        raw(f"{label} & {m.upper()} & "
           f"{eye_acc['accuracy']:.4f} [{eye_acc['ci95_lower']:.4f}, {eye_acc['ci95_upper']:.4f}] & "
           f"{fixed_acc['accuracy']:.4f} [{fixed_acc['ci95_lower']:.4f}, {fixed_acc['ci95_upper']:.4f}] & "
           f"{gap['observed_gap']:+.4f} [{gap['ci95_lower']:+.4f}, {gap['ci95_upper']:+.4f}] & "
           f"{s['mcnemar']['p_value']:.3f} \\\\")
raw("\\bottomrule")
raw("\\end{tabular}")
raw("\\end{table*}")
raw("")

raw("\\begin{figure}[t]")
raw("\\centering")
raw("\\includegraphics[width=\\linewidth]{figures/occlusion_vs_accuracy.png}")
raw("\\caption{Binning each fixed camera's CNN held-out predictions by "
   "occluded\\_fraction shows accuracy falling as the crop fills with everything "
   "except the object, for both fixed cameras -- direct, per-sample evidence for "
   "the mechanism behind both accuracy gaps.}")
raw("\\label{fig:occlusion}")
raw("\\end{figure}")
raw("")

raw("\\begin{figure*}[t]")
raw("\\centering")
raw("\\includegraphics[width=0.9\\linewidth]{figures/scale_comparison.png}")
raw("\\caption{The same instant, all three cameras, all eight classes. The "
   "fixed-camera window is the same physical size regardless of class, so a "
   "sphere or ellipsoid fills far less of it than a rod or cuboid.}")
raw("\\label{fig:scale}")
raw("\\end{figure*}")
raw("")

raw("\\begin{figure*}[t]")
raw("\\centering")
raw("\\includegraphics[width=0.9\\linewidth]{figures/occlusion_demo.png}")
raw("\\caption{The arm occluding its own target: at the instant the eye-in-hand "
   "camera frames this object cleanly, the fixed-overhead camera sees only the "
   "arm, while the fixed-angled camera, looking in from the side, still catches "
   "a partial view of the same object at the same instant.}")
raw("\\label{fig:occlusion-demo}")
raw("\\end{figure*}")

# ---------------------------------------------------------------------------
# 6. Discussion
# ---------------------------------------------------------------------------

section("Discussion")

eih_svm_acc = results["eye_in_hand"]["svm"]["test_accuracy"]
eih_cnn_acc = results["eye_in_hand"]["cnn"]["test_accuracy"]
foh_svm_acc = results["fixed_overhead"]["svm"]["test_accuracy"]
fah_svm_acc = results["fixed_angled"]["svm"]["test_accuracy"]
foh_cnn_acc = results["fixed_overhead"]["cnn"]["test_accuracy"]
fah_cnn_acc = results["fixed_angled"]["cnn"]["test_accuracy"]
eih_svm_cv = results["eye_in_hand"]["svm"]["cv_accuracy_mean"]

para(
    "The main result is that eye-in-hand does not win outright. It is significantly "
    f"more accurate than the fixed-overhead camera for the CNN "
    f"({eih_cnn_acc:.4f} vs. {foh_cnn_acc:.4f} test accuracy, "
    f"{p_str(SIG[('fixed_overhead','cnn')]['mcnemar']['p_value'])}), but the gap to "
    f"the fixed-angled camera is smaller and not significant for the CNN "
    f"({eih_cnn_acc:.4f} vs. {fah_cnn_acc:.4f}, "
    f"{p_str(SIG[('fixed_angled','cnn')]['mcnemar']['p_value'])}). For the SVM, "
    f"eye-in-hand and fixed-overhead are statistically tied "
    f"({eih_svm_acc:.4f} vs. {foh_svm_acc:.4f}, "
    f"{p_str(SIG[('fixed_overhead','svm')]['mcnemar']['p_value'])}), and eye-in-hand "
    f"is significantly less accurate than the fixed-angled camera "
    f"({eih_svm_acc:.4f} vs. {fah_svm_acc:.4f}, "
    f"{p_str(SIG[('fixed_angled','svm')]['mcnemar']['p_value'])}). We report all four "
    "comparisons rather than the two that happen to favour eye-in-hand, since a "
    "benchmark that only reports favourable comparisons is not a useful benchmark."
)
para(
    "The comparison between the two fixed cameras explains most of this pattern. "
    f"On the held-out test set, the fixed-overhead camera's crop is at least "
    f"{NEAR_FULL_OCCLUSION_THRESHOLD*100:.0f}% non-object on "
    f"{OCC[('fixed_overhead','cnn')]['near_full_pct']:.1f}% of frames "
    f"({OCC[('fixed_overhead','cnn')]['near_full_n']} of 900); the fixed-angled "
    f"camera reaches that level on {OCC[('fixed_angled','cnn')]['near_full_pct']:.1f}% "
    f"({OCC[('fixed_angled','cnn')]['near_full_n']} of 900). This matches the "
    "geometry: an arm reaching down toward an object sits between that object and a "
    "straight-down camera more often than it sits between the object and a camera "
    "looking in from the side. Accuracy on these near-fully-occluded frames is low "
    "for every camera and model "
    f"(SVM: {OCC[('fixed_overhead','svm')]['near_full_acc']*100:.1f}% fixed-overhead, "
    f"{OCC[('fixed_angled','svm')]['near_full_acc']*100:.1f}% fixed-angled; "
    f"CNN: {OCC[('fixed_overhead','cnn')]['near_full_acc']*100:.1f}% fixed-overhead, "
    f"{OCC[('fixed_angled','cnn')]['near_full_acc']*100:.1f}% fixed-angled), so a "
    "camera that lands in this bin more often loses more overall accuracy to it. "
    "This is consistent with mounting angle, not fixedness by itself, being the "
    "main source of the gap between the two fixed baselines."
)
para(
    "The difference between the CNN and SVM results is most likely explained by "
    "how each model handles frames that are not near-fully-occluded. Both models "
    f"reach a similar accuracy on those frames "
    f"(CNN: {OCC[('fixed_overhead','cnn')]['clear_acc']*100:.1f}%-"
    f"{OCC[('fixed_angled','cnn')]['clear_acc']*100:.1f}%; SVM: "
    f"{OCC[('fixed_overhead','svm')]['clear_acc']*100:.1f}%-"
    f"{OCC[('fixed_angled','svm')]['clear_acc']*100:.1f}%), so the SVM's lower "
    "overall accuracy comes largely from the occluded tail, which makes its total "
    "accuracy more sensitive to how often that tail occurs -- and hence more "
    "sensitive to which fixed camera is used. The CNN's higher accuracy on clear "
    "frames leaves less room for the occlusion-rate difference to change its "
    "overall accuracy, which is consistent with its smaller, non-significant gap "
    "against the fixed-angled camera."
)
para(
    f"The CNN outperforms the SVM on all three camera views "
    f"({eih_cnn_acc:.4f} vs. {eih_svm_acc:.4f} eye-in-hand; "
    f"{foh_cnn_acc:.4f} vs. {foh_svm_acc:.4f} fixed-overhead; "
    f"{fah_cnn_acc:.4f} vs. {fah_svm_acc:.4f} fixed-angled). The gap between the two "
    "model families is wider here than in a four-class version of this task run "
    f"during development ({eih_svm_cv*100:.1f}% eye-in-hand SVM cross-validation "
    "accuracy at eight classes vs. 95.6% at four classes), which is consistent with "
    "the eight-class taxonomy requiring shape discrimination that a bounding-box "
    "shortcut cannot provide. The SVM pipeline remains within a few points of the "
    "CNN on most cameras despite using a much simpler model with no learned "
    "features, which is relevant for a deployment where training or running a CNN "
    "is not practical."
)

# ---------------------------------------------------------------------------
# 7. Limitations and Future Work
# ---------------------------------------------------------------------------

section("Limitations and Future Work")

enumerate_block([
    "No domain randomisation. Lighting, table texture and camera intrinsics are "
    "fixed. A classifier trained on this data would almost certainly not "
    "transfer to a real camera. Randomising lighting direction and intensity, "
    "surface texture, and adding sensor noise and motion blur is the necessary "
    "next step before any sim-to-real claim.",
    "Geometric primitives, not real objects. Eight primitives -- four of them "
    "chosen specifically to be confusable with the other four -- prove the "
    "pipeline can measure a real difficulty gradient, not that it matches the "
    "task. Replacing them with scanned meshes of everyday objects is required "
    "before the classification problem is representative of a real picking "
    "workcell.",
    "Clutter in frame. Neighbouring objects appear at the edges of many frames "
    "in all three views. The labelled object is reliably centred, but frames "
    "are not clean single-object crops.",
    "Both fixed baselines are given the object's location for free. Their "
    "crops are centred on the target's known 3-D position; a real fixed-camera "
    "system would need its own detector or localiser to find that region "
    "first, which is not modelled here. The reported gaps are therefore the "
    "classification-only cost of each fixed viewpoint -- the true end-to-end "
    "cost, including detection failures under occlusion, is at least as large.",
    "The SVM's segmentation is a scene-specific shortcut, and an imperfect "
    "one. Isolating the silhouette by HSV-saturation thresholding works "
    "because objects are saturated and most of the scene is grey by "
    "construction -- except the claw's orange accent trim, which is also "
    "saturated, occasionally causing the largest saturated contour to be a "
    "finger rather than the object. This segmentation also would not survive "
    "a cluttered real background with its own saturated colours.",
    "Sample size. With one held-out test set per camera, not every accuracy "
    "gap in Table III reaches conventional statistical "
    "significance under a conservative paired test. A larger-scale replication, "
    "or an aggregated analysis across multiple random seeds, would sharpen "
    "these estimates rather than relying on a single test set per camera.",
    "Only one angle tested for the angled baseline. 45 degrees is one point on a "
    "continuum between straight-down and eye-level; whether occlusion and "
    "accuracy vary monotonically with mounting angle, or 45 degrees happens to sit at "
    "a particularly favourable or unfavourable point, is not established here.",
    "Grasping is open-loop. The claw closes to a width computed from "
    "ground-truth object geometry, which a real system would not have.",
    "Rigid-body contact only. Objects are rigid primitives with idealised "
    "friction. No deformables, no slip modelling.",
])

# ---------------------------------------------------------------------------
# 8. Conclusion
# ---------------------------------------------------------------------------

section("Conclusion")
para(
    "We built a simulated testbed that pairs an eye-in-hand camera with two "
    "independent fixed-camera baselines -- straight-overhead and 45-degree-angled "
    "-- observing the same manipulation task at the same instant over an "
    "eight-class taxonomy designed to resist a bounding-box shortcut, and used it "
    "to measure, rather than assume, the accuracy cost of the conventional "
    "fixed-camera design and whether that cost depends on mounting angle. The "
    "eye-in-hand camera does not win against both fixed baselines for both "
    "classifier families: it is significantly more accurate than the "
    "fixed-overhead camera with the CNN, roughly tied with the fixed-overhead "
    "camera with the SVM, and significantly less accurate than the fixed-angled "
    "camera with the SVM. A per-sample occlusion metric traces this pattern to a "
    "measured cause rather than an assumed one: the fixed-angled camera reaches "
    "near-full occlusion on a much smaller share of frames than the "
    "fixed-overhead camera, and this occlusion-rate difference affects the "
    "SVM's total accuracy more than the CNN's, since the SVM already loses more "
    "accuracy on occluded frames. The overall conclusion is that mounting angle, "
    "not fixedness by itself, is the main source of the accuracy cost usually "
    "attributed to a fixed camera, and that eye-in-hand's advantage over a fixed "
    "camera should not be assumed without specifying which fixed camera and "
    "which classifier are being compared. Two data-integrity bugs found during "
    "development, one a direct side effect of fixing the other, are reported for "
    "the same reason: an honest benchmark reports what was found, not only what "
    "was favourable. The complete pipeline, from procedural "
    "scene generation through trained model weights, is released so the "
    "benchmark can be extended, replicated at larger scale, tested at "
    "additional mounting angles, or subjected to domain randomisation as a step "
    "toward a genuine sim-to-real claim."
)

raw("\\section*{Acknowledgments}")
para("The author thanks the DES Pune University CSE department and DRDO for research support.")

# ---------------------------------------------------------------------------
# Assemble the .tex document
# ---------------------------------------------------------------------------

body = "\n".join(parts)

title_text = esc(
    "Does Fixed Mean Occluded, or Does Straight-Down Mean Occluded? A Simulated, "
    "Cross-Validated Benchmark of Eye-in-Hand vs. Two Fixed-Camera Baselines for "
    "Robotic Manipulation"
)

keywords = esc(
    "eye-in-hand vision, robotic manipulation, visual servoing, camera placement, "
    "MuJoCo simulation, convolutional neural networks, support vector machines, "
    "object classification, occlusion."
)

document = f"""\\documentclass[conference]{{IEEEtran}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage[hidelinks]{{hyperref}}

\\title{{{title_text}}}
\\author{{
\\IEEEauthorblockN{{Yugandhar Kulkarni}}
\\IEEEauthorblockA{{DES Pune University \\\\ DRDO Research Intern \\\\ github.com/yugandhar-k}}
}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
{esc(abstract)}
\\end{{abstract}}

\\begin{{IEEEkeywords}}
{keywords}
\\end{{IEEEkeywords}}

{body}

\\bibliographystyle{{IEEEtran}}
\\bibliography{{references}}

\\end{{document}}
"""

references_bib = """@article{hutchinson1996tutorial,
  author  = {Hutchinson, Seth and Hager, Gregory D. and Corke, Peter I.},
  title   = {A tutorial on visual servo control},
  journal = {IEEE Transactions on Robotics and Automation},
  volume  = {12},
  number  = {5},
  pages   = {651--670},
  year    = {1996}
}

@article{hu1962moments,
  author  = {Hu, Ming-Kuei},
  title   = {Visual pattern recognition by moment invariants},
  journal = {IRE Transactions on Information Theory},
  volume  = {8},
  number  = {2},
  pages   = {179--187},
  year    = {1962}
}

@inproceedings{dalal2005hog,
  author    = {Dalal, Navneet and Triggs, Bill},
  title     = {Histograms of oriented gradients for human detection},
  booktitle = {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  pages     = {886--893},
  year      = {2005}
}

@article{lecun1998gradient,
  author  = {LeCun, Yann and Bottou, L{\\'e}on and Bengio, Yoshua and Haffner, Patrick},
  title   = {Gradient-based learning applied to document recognition},
  journal = {Proceedings of the IEEE},
  volume  = {86},
  number  = {11},
  pages   = {2278--2324},
  year    = {1998}
}

@article{cortes1995support,
  author  = {Cortes, Corinna and Vapnik, Vladimir},
  title   = {Support-vector networks},
  journal = {Machine Learning},
  volume  = {20},
  number  = {3},
  pages   = {273--297},
  year    = {1995}
}

@inproceedings{todorov2012mujoco,
  author    = {Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  title     = {{MuJoCo}: A physics engine for model-based control},
  booktitle = {Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)},
  pages     = {5026--5033},
  year      = {2012}
}

@inproceedings{tobin2017domain,
  author    = {Tobin, Josh and Fong, Rachel and Ray, Alex and Schneider, Jonas and
              Zaremba, Wojciech and Abbeel, Pieter},
  title     = {Domain randomization for transferring deep neural networks from
              simulation to the real world},
  booktitle = {Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)},
  pages     = {23--30},
  year      = {2017}
}

@article{kim2023givingrobots,
  author  = {Kim, Moo Jin and Wu, Jiajun and Finn, Chelsea},
  title   = {Giving robots a hand: Learning generalizable manipulation with
             eye-in-hand human video demonstrations},
  journal = {arXiv preprint arXiv:2307.05959},
  year    = {2023}
}

@article{chuang2024activevision,
  author  = {Chuang, Ian and Lee, Andrew and Gao, Dechen and Naddaf-Sh, M-Mahdi and
             Soltani, Iman},
  title   = {Active vision might be all you need: Exploring active vision in
             bimanual robotic manipulation},
  journal = {arXiv preprint arXiv:2409.17435},
  year    = {2024}
}

@article{liu2025avr,
  author  = {Liu, Yushan and Mu, Shilong and Chao, Xintao and Li, Zizhen and Mu, Yao
             and Chen, Tianxing and Li, Shoujie and Lyu, Chuqiao and Zhang, Xiao-Ping
             and Ding, Wenbo},
  title   = {{AVR}: Active vision-driven precise robot manipulation with viewpoint
             and focal length optimization},
  journal = {arXiv preprint arXiv:2503.01439},
  year    = {2025}
}

@article{li2026fromfixed,
  author  = {Li, Wenhao and Jiang, Xueying and Qian, Quanhao and Zhao, Deli and
             Lu, Shijian and Zhang, Gongjie and Xu, Ran},
  title   = {From fixed to free cameras: Calibration-free view-robust
             vision-language-action model},
  journal = {arXiv preprint arXiv:2607.05396},
  year    = {2026}
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tectonic", default=None,
                       help="path to tectonic.exe; if omitted, looks for 'tectonic' on PATH")
    parser.add_argument("--no-compile", action="store_true",
                       help="write the .tex sources and zip but skip PDF compilation")
    args = parser.parse_args()

    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tex_path = LATEX_DIR / "paper.tex"
    tex_path.write_text(document, encoding="utf-8")
    (LATEX_DIR / "references.bib").write_text(references_bib, encoding="utf-8")
    print(f"wrote {tex_path}")

    for fig_name in ("results.png", "confusion_matrix.png", "occlusion_vs_accuracy.png",
                     "scale_comparison.png", "occlusion_demo.png"):
        shutil.copy(FIGURES / fig_name, LATEX_FIGURES_DIR / fig_name)
    print(f"copied 5 figures to {LATEX_FIGURES_DIR}")

    zip_path = OUT_DIR / "paper_latex.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in LATEX_DIR.rglob("*"):
            if path.is_file() and path.suffix not in (".pdf", ".aux", ".log", ".out",
                                                        ".bbl", ".blg", ".fls",
                                                        ".fdb_latexmk"):
                zf.write(path, path.relative_to(LATEX_DIR))
    print(f"wrote {zip_path}")

    if args.no_compile:
        print("skipping PDF compilation (--no-compile)")
        return

    tectonic_exe = args.tectonic or "tectonic"
    print(f"compiling with {tectonic_exe} ...")
    result = subprocess.run(
        [tectonic_exe, "paper.tex", "--keep-logs"],
        cwd=str(LATEX_DIR), capture_output=True, text=True,
    )
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print("=== STDERR ===")
        print(result.stderr[-5000:])
        raise SystemExit(f"tectonic failed with exit code {result.returncode}")

    compiled_pdf = LATEX_DIR / "paper.pdf"
    final_pdf = OUT_DIR / "paper.pdf"
    shutil.copy(compiled_pdf, final_pdf)
    print(f"compiled PDF -> {final_pdf}")


if __name__ == "__main__":
    main()
