"""demo.py - Live, interactive demo of the eye-in-hand vs. both fixed-camera baselines.

The robot runs its usual inspect-and-pick cycle. At any moment you can switch which
camera is feeding the classifier (the eye-in-hand claw camera, the fixed-overhead
baseline, or the fixed-angled baseline) and which trained model is doing the classifying
(the CNN or the HOG+Hu+SVM pipeline) - the same six checkpoints `src.train` produced, run
on exactly the preprocessing they were trained with. Nothing here is a re-enactment:
every prediction is a live forward pass through the actual trained model on the actual
current camera frame.

Requires the benchmark to have been run first (`python run_sim.py --dataset ...` then
the six `python -m src.train ...` calls - see the top-level README's
"Classification Benchmark" section) so that `checkpoints/*_cnn.pt` and
`checkpoints/*_svm.joblib` exist for all three views.

Usage
-----
    python demo.py
    python demo.py --seed 7

Keys
----
  SPACE   pause / resume
  V       switch camera: eye-in-hand -> fixed-overhead -> fixed-angled -> ...
  N       switch model: CNN <-> SVM
  G       grasp on / off
  R       reshuffle the object layout
  +/-     speed up / slow down
  ESC     quit

Author: Yugandhar Kulkarni - DRDO Research Intern - DES Pune University
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Optional, Tuple

import glfw
import joblib
import mujoco
import numpy as np
import torch
from PIL import Image

import run_sim as sim
from src.dataset import extract_shape_features
from src.model import CompactCNN
from src.utils import (
    CHECKPOINT_DIR,
    CLASS_NAMES,
    CNN_IMAGE_SIZE,
    FIXED_VIEWS,
    NUM_CLASSES,
    RANDOM_SEED,
    VIEWS,
)

VIEW_LABELS: Dict[str, str] = {
    "eye_in_hand": "EYE-IN-HAND",
    "fixed_overhead": "FIXED-OVERHEAD",
    "fixed_angled": "FIXED-ANGLED",
}
# The MuJoCo camera each view reads from - see run_sim.build_mjcf.
VIEW_CAMERA_NAME: Dict[str, str] = {
    "eye_in_hand": "claw",
    "fixed_overhead": "overhead",
    "fixed_angled": "angled",
}
MODELS: Tuple[str, str] = ("cnn", "svm")

# Must match src.dataset.SimImageDataset - the CNN was trained against these stats.
_CNN_MEAN = np.array((0.5, 0.5, 0.5), dtype=np.float32)
_CNN_STD = np.array((0.22, 0.22, 0.22), dtype=np.float32)
_SVM_FEATURE_SIZE = 128    # matches src.dataset.build_feature_matrix's load_rgb size


class Predictor:
    """Loads all four (view, model) checkpoints once; predicts on demand.

    Every prediction is a live forward pass through the checkpoint `src.train` produced
    for that exact (view, model) pair, on preprocessing that matches training as closely
    as a single live frame allows (see each predict method's docstring for the one place
    it necessarily differs: model input size vs. the saved-PNG round-trip training used).
    """

    def __init__(self) -> None:
        missing = [
            path for view in VIEWS
            for path in (CHECKPOINT_DIR / f"{view}_cnn.pt", CHECKPOINT_DIR / f"{view}_svm.joblib")
            if not path.exists()
        ]
        if missing:
            names = "\n  ".join(str(p) for p in missing)
            raise FileNotFoundError(
                f"missing checkpoint(s):\n  {names}\n"
                "run the six `python -m src.train --view ... --model ...` commands "
                "first (see README -> Classification Benchmark)."
            )
        self._cnn = {view: self._load_cnn(view) for view in VIEWS}
        self._svm = {view: joblib.load(CHECKPOINT_DIR / f"{view}_svm.joblib")
                    for view in VIEWS}

    @staticmethod
    def _load_cnn(view: str) -> CompactCNN:
        model = CompactCNN(NUM_CLASSES)
        state = torch.load(CHECKPOINT_DIR / f"{view}_cnn.pt", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    @torch.no_grad()
    def _predict_cnn(self, view: str, rgb: np.ndarray) -> Tuple[str, float]:
        """Resize to `CNN_IMAGE_SIZE`, normalise, forward pass, softmax."""
        image = Image.fromarray(rgb).resize((CNN_IMAGE_SIZE, CNN_IMAGE_SIZE), Image.LANCZOS)
        array = np.asarray(image).astype(np.float32) / 255.0
        array = (array - _CNN_MEAN) / _CNN_STD
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float().unsqueeze(0)
        probs = torch.softmax(self._cnn[view](tensor), dim=1)[0]
        class_id = int(probs.argmax())
        return CLASS_NAMES[class_id], float(probs[class_id])

    def _predict_svm(self, view: str, rgb: np.ndarray) -> Tuple[str, float]:
        """Resize to the training feature size, extract HOG+Hu features, classify.

        The SVM was fit without `probability=True` (see `src.model.build_svm_pipeline`'s
        docstring for why), so it has no calibrated class probabilities. The number
        shown is a softmax over its one-vs-rest decision-function margins - a rough,
        uncalibrated indicator of how confidently separated the winning class is, not a
        probability. It is presented as such in the on-screen readout.
        """
        image = Image.fromarray(rgb).resize((_SVM_FEATURE_SIZE, _SVM_FEATURE_SIZE),
                                            Image.LANCZOS)
        features = extract_shape_features(np.asarray(image))[None, :]
        pipeline = self._svm[view]
        scores = pipeline.decision_function(features)[0]
        exp_scores = np.exp(scores - scores.max())
        weights = exp_scores / exp_scores.sum()
        class_id = int(np.argmax(scores))
        return CLASS_NAMES[class_id], float(weights[class_id])

    def predict(self, model_name: str, view: str, rgb: np.ndarray) -> Tuple[str, float]:
        if model_name == "cnn":
            return self._predict_cnn(view, rgb)
        return self._predict_svm(view, rgb)


class LiveCameras:
    """Renders all three camera views headlessly, matching `run_sim.generate_dataset`
    exactly - one `SimCamera` per view, keyed the same way `VIEW_CAMERA_NAME` is.
    """

    def __init__(self, workspace: sim.Workspace) -> None:
        self.workspace = workspace
        self._cameras = {
            view: sim.SimCamera(workspace, camera_name)
            for view, camera_name in VIEW_CAMERA_NAME.items()
        }

    def eye_in_hand_frame(self) -> np.ndarray:
        return self._cameras["eye_in_hand"].render_rgb()

    def fixed_view_frame(self, view: str, target: sim.SceneObject
                         ) -> Tuple[np.ndarray, Tuple[float, float, float, int]]:
        """Full frame from `view`'s fixed camera, cropped around `target`'s known
        position - the same fixed, class-independent crop size used for both fixed
        cameras during dataset generation (`run_sim.FIXED_CAMERA_CROP_HALF_SIDE`).

        Returns:
            (crop, crop_box) where crop_box is (centre_x, centre_y, half_side,
            source_size) in that camera's own pixel space - passed straight through to
            `PictureInPictureViewer.render`'s `crop_box` so the on-screen panel can
            outline exactly this region.
        """
        camera = self._cameras[view]
        rgb = camera.render_rgb()
        pixel_x, pixel_y, depth = camera.project(target.position)
        half_side_px = sim.FIXED_CAMERA_CROP_HALF_SIDE * camera.focal_px / depth
        crop = sim.crop_square(rgb, pixel_x, pixel_y, half_side_px, sim.CAM_IMAGE_SIZE)
        return crop, (pixel_x, pixel_y, half_side_px, sim.CAM_IMAGE_SIZE)

    def crop_box_for(self, view: str, target: sim.SceneObject
                     ) -> Tuple[float, float, float, int]:
        """Just the crop-box geometry for `view`, without re-rendering the frame -
        used to keep the on-screen outline drawn between inference ticks.
        """
        camera = self._cameras[view]
        pixel_x, pixel_y, depth = camera.project(target.position)
        half_side_px = sim.FIXED_CAMERA_CROP_HALF_SIDE * camera.focal_px / depth
        return (pixel_x, pixel_y, half_side_px, sim.CAM_IMAGE_SIZE)

    def close(self) -> None:
        for camera in self._cameras.values():
            camera.close()


def run_demo(seed: int) -> None:
    print("[demo] loading trained checkpoints ...")
    predictor = Predictor()

    workspace = sim.Workspace(seed)
    kinematics = workspace.kinematics
    scan = sim.ScanController(workspace, grasping=True)
    cameras = LiveCameras(workspace)
    viewer = sim.PictureInPictureViewer(
        workspace, "Eye-in-Hand vs. Fixed Cameras - Live Classification Demo"
    )
    print("[demo] window opened - titled 'Eye-in-Hand vs. Fixed Cameras - Live "
          "Classification Demo'. If you don't see it, check other monitors/desktops "
          "and the taskbar; alt-tab should find it.")

    state = {
        "paused": False,
        "speed": 1.0,
        "seed": seed,
        "view_index": 0,
        "model_index": 0,
        "correct": 0,
        "total": 0,
        "last_prediction": "-",
        "last_confidence": 0.0,
        "last_truth": "-",
        "last_frame_index": -1,
    }

    def current_view() -> str:
        return VIEWS[state["view_index"] % len(VIEWS)]

    def current_model() -> str:
        return MODELS[state["model_index"] % len(MODELS)]

    def reset_tally() -> None:
        state["correct"] = 0
        state["total"] = 0

    def on_key(key: int) -> None:
        if key == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
        elif key == glfw.KEY_V:
            state["view_index"] += 1
            reset_tally()
        elif key == glfw.KEY_N:
            state["model_index"] += 1
            reset_tally()
        elif key == glfw.KEY_G:
            scan.grasping = not scan.grasping
            scan.begin(time.time(), index=scan.index)
        elif key == glfw.KEY_R:
            state["seed"] += 1
            workspace.reshuffle(state["seed"])
            kinematics._last_solution = None
            scan.begin(time.time(), index=0)
            reset_tally()
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            state["speed"] = min(4.0, state["speed"] * 1.25)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            state["speed"] = max(0.2, state["speed"] / 1.25)

    viewer.key_handler = on_key

    print(f"[demo] seed {seed} | classes: {', '.join(c.name for c in sim.OBJECT_CLASSES)}")
    print(f"[demo] cameras (V cycles through): {', '.join(VIEWS)}")
    print("[demo] keys: SPACE pause | V switch camera | N switch model | G grasp on/off")
    print("[demo]       R reshuffle | +/- speed | ESC quit")
    print("[demo] the SVM's on-screen number is an uncalibrated decision-margin score, "
          "not a probability - the CNN's is a true softmax probability.")

    scan.begin(time.time())
    step_index = 0
    inference_countdown = 0
    INFERENCE_EVERY_N_FRAMES = 5      # ~12 Hz at 60 fps, plenty responsive, low load

    try:
        while viewer.is_running():
            frame_start = time.time()

            if not state["paused"]:
                for _ in range(int(round((1.0 / 60.0) / sim.SIM_TIMESTEP))):
                    if step_index % sim.CONTROL_DECIMATION == 0:
                        command, _ = scan.update(time.time(), state["speed"])
                        angles = kinematics.solve(workspace.data, command.position,
                                                  command.yaw)
                        workspace.command(angles, command.gap)
                    mujoco.mj_step(workspace.model, workspace.data)
                    step_index += 1

            view = current_view()
            model_name = current_model()
            target = workspace.objects[scan.index % len(workspace.objects)]
            crop_box: Optional[Tuple[float, float, float, int]] = None

            inference_countdown -= 1
            if inference_countdown <= 0 and not state["paused"]:
                inference_countdown = INFERENCE_EVERY_N_FRAMES
                if view == "eye_in_hand":
                    frame = cameras.eye_in_hand_frame()
                else:
                    frame, crop_box = cameras.fixed_view_frame(view, target)

                predicted_name, confidence = predictor.predict(model_name, view, frame)
                truth_name = target.object_class.name
                if scan.index != state["last_frame_index"] and scan.phase == "settle":
                    # Only tally once per object, at the same decision point training
                    # used, so the running accuracy is comparable to the paper's numbers.
                    state["total"] += 1
                    state["correct"] += int(predicted_name == truth_name)
                    state["last_frame_index"] = scan.index
                state["last_prediction"] = predicted_name
                state["last_confidence"] = confidence
                state["last_truth"] = truth_name
            elif view in FIXED_VIEWS:
                # Keep the crop box drawn between inference ticks so it doesn't flicker.
                crop_box = cameras.crop_box_for(view, target)

            if view == "eye_in_hand":
                viewer.set_panel_camera("claw", "EYE-IN-HAND CAMERA (model input)")
            else:
                viewer.set_panel_camera(
                    VIEW_CAMERA_NAME[view],
                    f"{VIEW_LABELS[view]} CAMERA (green box = model input)",
                )

            correct_marker = "?"
            if state["last_prediction"] != "-":
                correct_marker = "OK" if state["last_prediction"] == state["last_truth"] \
                    else "WRONG"
            accuracy = (100.0 * state["correct"] / state["total"]) if state["total"] else 0.0
            score_label = "confidence" if model_name == "cnn" else "score (uncalibrated)"

            viewer.render(
                overlay_lines=(
                    (f"camera: {VIEW_LABELS[view]}   model: {model_name.upper()}"
                     f"{'  [PAUSED]' if state['paused'] else ''}\n"
                     f"phase: {scan.phase.upper()}\n"
                     f"prediction: {state['last_prediction']}  "
                     f"({score_label}: {state['last_confidence']*100:.0f}%)   "
                     f"ground truth: {state['last_truth']}   [{correct_marker}]\n"
                     f"session accuracy: {state['correct']}/{state['total']} "
                     f"({accuracy:.1f}%)",
                     ""),
                    ("SPACE pause   V camera   N model   G grasp   R reshuffle   "
                     "+/- speed   ESC quit",
                     ""),
                ),
                crop_box=crop_box,
            )

            remaining = (1.0 / 60.0) - (time.time() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        viewer.close()
        cameras.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="object layout seed")
    args = parser.parse_args()
    run_demo(args.seed)


if __name__ == "__main__":
    main()
