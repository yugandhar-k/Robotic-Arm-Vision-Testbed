"""perception.py - Vision-only object detection and localisation for
search_and_pick.py.

Every function here works from a rendered camera frame (RGB + depth + segmentation) and
the camera's own extrinsics (`SimCamera.project`/`unproject`, always read from wherever
the arm actually is right now) - never from `SceneObject.position` or any other
per-object ground truth. What *is* used, and is legitimate, is the robot's own static
self/environment model: which geoms belong to a free-floating rigid body (a candidate
object) versus the robot, table or basket (all fixed or hinge-jointed, hence never a pick
target). A real system gets the same distinction for free from its own URDF plus known,
fixed workspace geometry - it is not privileged information about where anything sits.

Detection therefore reduces to "which pixels belong to some free body, and where is the
biggest blob of them" - a simulated stand-in for depth-based background subtraction
(e.g. RANSAC table-plane removal + Euclidean clustering), which is the standard technique
real bin-picking systems use to find candidate objects, deliberately different from the
colour-saturation heuristic the classification benchmark's SVM pipeline uses downstream -
depth/geometry, not colour, is what should decide "is this an object" here, since colour
segmentation is exactly what confused the claw's own (saturated) finger pads for the
target object during development (see README limitations).

Author: Yugandhar Kulkarni - DRDO Research Intern - DES Pune University
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

import cv2
import mujoco
import numpy as np

from run_sim import SimCamera

MIN_CONTOUR_AREA_PX: float = 40.0    # ignore specks (sensor noise, stray pixels)
YAW_PROBE_EPS: float = 0.02          # world-metre perturbation used to calibrate angle


@dataclass
class Detection:
    """One vision-estimated object pose, in world coordinates - the perception
    pipeline's entire output. Nothing downstream of this needs to know which camera, or
    which technique, produced it.
    """

    position: np.ndarray       # (3,) world xyz of the object's visible top surface
    yaw: float                 # world yaw [rad], mod pi, to align the claw's closing axis
    width_m: float              # estimated width across the closing direction [m]
    pixel_area: float           # contour area in the source frame [px^2], for logging
    pixel_centroid: Tuple[float, float]


def free_body_geom_ids(model: mujoco.MjModel) -> FrozenSet[int]:
    """Geom ids belonging to a free-floating body - i.e. a possible pick target.

    Computed once from the model's own joint structure (which bodies have a 6-DoF free
    joint, as every object does and no arm link, table or basket does) - never from any
    body's current position. Stands in for a robot's own static self/environment model:
    "this geometry is mine (or bolted down); anything else in view is a candidate."
    """
    free_body_ids = {
        body_id for body_id in range(model.nbody)
        if model.body_jntnum[body_id] == 1
        and model.jnt_type[model.body_jntadr[body_id]] == mujoco.mjtJoint.mjJNT_FREE
    }
    return frozenset(
        geom_id for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] in free_body_ids
    )


def _masked_depth(depth: np.ndarray, mask: np.ndarray) -> float:
    """Median depth under a boolean pixel mask - robust to a few noisy edge pixels."""
    values = depth[mask]
    return float(np.median(values)) if values.size else float(np.median(depth))


def _principal_axis_angle(mask: np.ndarray) -> float:
    """Image-plane orientation of a binary mask's longest axis, from its central moments.

    theta = 0.5 * atan2(2*mu11, mu20 - mu02) is the closed form for a blob's major-axis
    angle. Defined mod pi (an axis, not a direction), which is exactly what a two-finger
    claw needs.
    """
    moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    return 0.5 * math.atan2(2.0 * moments["mu11"], moments["mu20"] - moments["mu02"])


def _image_angle_to_world_yaw(camera: SimCamera, world_point: np.ndarray,
                              image_angle_rad: float,
                              eps: float = YAW_PROBE_EPS) -> float:
    """Convert an image-plane orientation angle into a world yaw about Z.

    Projects two small perturbations of `world_point` along world +X and +Y to learn,
    numerically, how this camera's pixel axes relate to world axes at this instant -
    correct for any camera pose (straight-down, 45-degree, tilted eye-in-hand) without
    hand-deriving a sign convention per camera.
    """
    px0, py0, _ = camera.project(world_point)
    px_x, py_x, _ = camera.project(world_point + np.array([eps, 0.0, 0.0]))
    px_y, py_y, _ = camera.project(world_point + np.array([0.0, eps, 0.0]))
    image_dir_x = np.array([px_x - px0, py_x - py0])
    image_dir_y = np.array([px_y - px0, py_y - py0])
    basis = np.stack([image_dir_x, image_dir_y], axis=1)   # columns = world x, y images
    rect_dir = np.array([math.cos(image_angle_rad), math.sin(image_angle_rad)])
    coeffs = np.linalg.solve(basis, rect_dir)
    return math.atan2(coeffs[1], coeffs[0])


def _detection_from_contour(camera: SimCamera, contour: np.ndarray,
                            candidate: np.ndarray, depth_map: np.ndarray
                            ) -> Optional[Detection]:
    """Build one `Detection` from a single contour already known to be big enough.

    Shared by `detect` (one contour, from a centre-restricted search) and `detect_all`
    (every contour, from a whole, unrestricted frame) so both report the identical pose
    estimate for the same blob.
    """
    blob_mask = np.zeros(candidate.shape, dtype=np.uint8)
    cv2.drawContours(blob_mask, [contour], -1, color=1, thickness=-1)
    blob_mask = blob_mask.astype(bool) & candidate

    moments = cv2.moments(blob_mask.astype(np.uint8), binaryImage=True)
    if moments["m00"] == 0:
        return None
    centroid_x = moments["m10"] / moments["m00"]
    centroid_y = moments["m01"] / moments["m00"]

    depth = _masked_depth(depth_map, blob_mask)
    world_point = camera.unproject(centroid_x, centroid_y, depth)

    (_, _), (w_px, h_px), _ = cv2.minAreaRect(contour)
    metres_per_px = depth / camera.focal_px
    width_m = min(w_px, h_px) * metres_per_px

    axis_angle = _principal_axis_angle(blob_mask)
    yaw = _image_angle_to_world_yaw(camera, world_point, axis_angle + math.pi / 2.0)

    return Detection(
        position=world_point,
        yaw=yaw,
        width_m=width_m,
        pixel_area=float(cv2.contourArea(contour)),
        pixel_centroid=(centroid_x, centroid_y),
    )


def detect(camera: SimCamera, candidate_geom_ids: FrozenSet[int],
          min_area_px: float = MIN_CONTOUR_AREA_PX,
          patch_fraction: float = 0.55) -> Optional[Detection]:
    """Look through `camera` right now and report a candidate object, if any.

    Restricted to the centre `patch_fraction` of the frame - the same centre-patch
    convention `label_from_segmentation` uses for ground-truth labelling, applied here so
    a neighbouring object visible at the frame's edge (the object field's 130 mm slot
    spacing is well within the eye-in-hand camera's footprint at a normal hover height)
    cannot be mistaken for the one actually under inspection. Meant for a directed look
    (the eye-in-hand search sweep, or the confirm step) where exactly one candidate -
    whatever is centred - is expected.

    Returns None if nothing large enough to be an object is in view - the caller moves on
    and tries elsewhere.
    """
    segmentation = camera.render_segmentation()
    candidate = np.isin(segmentation, list(candidate_geom_ids))

    height, width = candidate.shape
    half = int(min(height, width) * patch_fraction / 2.0)
    centre_y, centre_x = height // 2, width // 2
    patch_mask = np.zeros_like(candidate)
    patch_mask[centre_y - half: centre_y + half, centre_x - half: centre_x + half] = True
    candidate &= patch_mask

    if not candidate.any():
        return None

    contours, _ = cv2.findContours(candidate.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area_px:
        return None

    depth_map = camera.render_depth()
    return _detection_from_contour(camera, contour, candidate, depth_map)


def detect_all(camera: SimCamera, candidate_geom_ids: FrozenSet[int],
              min_area_px: float = MIN_CONTOUR_AREA_PX) -> List[Detection]:
    """Look through `camera` right now and report every candidate object in the frame.

    Unlike `detect`, there is no centre-patch restriction: meant for a fixed, wide-view
    camera (`fixed_overhead`/`fixed_angled`) that can see the whole object field in one
    shot, where finding every object at once - not just whichever is centred - is exactly
    the point. Returned in descending order of pixel area (largest, least likely to be a
    partially-occluded sliver, first).
    """
    segmentation = camera.render_segmentation()
    candidate = np.isin(segmentation, list(candidate_geom_ids))
    if not candidate.any():
        return []

    contours, _ = cv2.findContours(candidate.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_enough = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    if not big_enough:
        return []
    big_enough.sort(key=cv2.contourArea, reverse=True)

    depth_map = camera.render_depth()
    detections = []
    for contour in big_enough:
        detection = _detection_from_contour(camera, contour, candidate, depth_map)
        if detection is not None:
            detections.append(detection)
    return detections


def masked_area_in_region(camera: SimCamera, candidate_geom_ids: FrozenSet[int],
                          center_world: np.ndarray, half_width_m: float,
                          max_height_m: float = 0.15) -> int:
    """Count candidate pixels within a known square world-space region, as seen by
    `camera` right now - e.g. "how much of the basket's footprint is currently filled".

    The region's centre and size are known, fixed infrastructure (the same knowledge the
    basket's own drop target already uses), not per-object information. Only the pixel
    footprint is restricted to the region; detection itself is still purely from the live
    segmentation buffer, so this reports what is really there right now, not what was
    ever placed there.

    Deliberately a raw pixel *area*, not a count of distinct objects: two objects resting
    close together in a small basket routinely touch or overlap in a single 2-D view and
    merge into one contour, undercounting badly (measured: four objects at the basket's
    own drop-offset spacing rendered as a single blob). A per-pick area *increase*,
    checked before and after one release (see `search_and_pick.SearchAndPlaceController`),
    is a far more reliable "did this specific drop actually add something" signal than
    trying to segment individual objects in a pile.

    Two ground planes (`center_world`'s own height, and `max_height_m` above it) are
    projected and their footprints unioned, so a camera looking in at an angle - where a
    taller pile of objects appears higher in the frame than the bare floor does - doesn't
    silently crop off anything stacked above the first item.
    """
    segmentation = camera.render_segmentation()
    candidate = np.isin(segmentation, list(candidate_geom_ids))
    if not candidate.any():
        return 0

    pixel_xs: List[float] = []
    pixel_ys: List[float] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (0.0, max_height_m):
                corner = center_world + np.array([sx * half_width_m, sy * half_width_m, sz])
                px, py, _ = camera.project(corner)
                pixel_xs.append(px)
                pixel_ys.append(py)

    height, width = candidate.shape
    x0, x1 = max(0, int(min(pixel_xs))), min(width, int(max(pixel_xs)) + 1)
    y0, y1 = max(0, int(min(pixel_ys))), min(height, int(max(pixel_ys)) + 1)
    if x0 >= x1 or y0 >= y1:
        return 0

    return int(candidate[y0:y1, x0:x1].sum())
