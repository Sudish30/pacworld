"""Metrics for generated vs ground-truth 64x64 frames and rollouts.

Per frame (model frame vs the ground-truth frame at the same step):
  pacman_error     distance in 64x64 pixels between detected Pac-Man centroids
  ghost_count      number of the four ghosts detected (0-4)
  pellet_iou       IoU of the present-pellet sets
  wall_iou         IoU of the wall mask against the static maze reference
Ground-truth gating from RAM:
  ghost_eligibility  frames that are not frightened, not in a death animation,
                     and not in the start-of-life pen period
Per rollout:
  action responsiveness: at steps where the ground-truth direction (ram[56] & 3)
  changes to a direction contained in the held action, whether the model's
  Pac-Man movement direction matches within `response_window` steps.
"""
import numpy as np

from detectors import GHOST_NAMES

# ram[56] & 3 -> (drow, dcol)
DIRS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
DIR_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
# ALE action -> set of direction codes it contains
ACTION_DIRS = {0: set(), 1: {0}, 2: {1}, 3: {3}, 4: {2}, 5: {0, 1}, 6: {0, 3}, 7: {2, 1}, 8: {2, 3}}


def pacman_error(det_model, det_gt):
    """(error in px or nan, model_detected, gt_detected)."""
    m, g = det_model.get("pac"), det_gt.get("pac")
    if m is None or g is None:
        return np.nan, m is not None, g is not None
    return float(np.hypot(m[0] - g[0], m[1] - g[1])), True, True


def ghost_count(det):
    return int(sum(det.get(g) is not None for g in GHOST_NAMES))


def pellet_iou(presence_model, presence_gt):
    union = (presence_model | presence_gt).sum()
    return float((presence_model & presence_gt).sum() / union) if union else 1.0


def ghost_eligibility(ram, cfg):
    """Boolean array over frames: RAM says ghosts should be visible in their normal colours."""
    g = cfg["gating"]
    T = len(ram)
    ok = ram[:, g["frightened_ram"]] == 0
    lives = ram[:, g["lives_ram"]].astype(int)
    ok[: g["start_of_life_steps"]] = False
    for t in np.nonzero(np.diff(lives) < 0)[0]:
        ok[max(0, t - g["death_anim_steps"]): t + g["start_of_life_steps"]] = False
    ok[max(0, T - g["death_anim_steps"]):] = False
    return ok


def gt_directions(ram, cfg):
    return ram[:, cfg["responsiveness"]["direction_ram"]].astype(int) & 3


def movement_direction(positions, k, cfg):
    """Direction code of the move ending at index k, from positions[k - smooth] to positions[k]; None if unknown."""
    r = cfg["responsiveness"]
    s = r["smooth_steps"]
    if k - s < 0 or positions[k] is None or positions[k - s] is None:
        return None
    dr, dc = positions[k][0] - positions[k - s][0], positions[k][1] - positions[k - s][1]
    n = np.hypot(dr, dc)
    if n < r["min_move_px"] or n > r["max_move_px"]:
        return None
    if abs(dr) >= abs(dc):
        return 2 if dr > 0 else 0
    return 1 if dc > 0 else 3


def responsiveness_events(gt_dir, actions, start, end):
    """Events (frame index i, new direction) for i in [start, end): GT direction changed at i to a
    direction contained in actions[i-1], the action that produced frame i."""
    events = []
    for i in range(max(start, 1), end):
        if gt_dir[i] != gt_dir[i - 1] and gt_dir[i] in ACTION_DIRS[int(actions[i - 1])]:
            events.append((i, int(gt_dir[i])))
    return events


def responsiveness_hits(events, positions, start, offset, cfg):
    """For each event at frame i, whether the model's movement direction equals the new direction at
    some index k in [i - start + offset, + response_window]. `positions` is indexed with `offset`
    context entries before the first predicted frame."""
    w = cfg["responsiveness"]["response_window"]
    hits = []
    for i, d in events:
        k0 = i - start + offset
        hits.append(any(movement_direction(positions, k, cfg) == d for k in range(k0, min(k0 + w + 1, len(positions)))))
    return hits
