# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Post-process TokenGS-compatible Gaussian PLY (e.g. NRE harvested assets).

Pipeline order: **upside-down flip → front-back flip → pivot → optional scale**. The pivot matches
AH ``save_ply`` in raw units first; metadata **uniform scale** is applied **last** with
``anchor='origin'`` (``p *= s``) so nothing re-centers the cloud and lifts the car off the ground.

1. **Upside-down** fix: first 180° rotation (``--flip-axis``, default x).
2. **Front-back**: second 180° rotation (``--also-flip-axis``, e.g. y).
3. **Ground-plane pivot**: bottom contact in Y; horizontal per ``--pivot-xz`` (center / rear / front).
4. Optional **uniform scale** to metadata cuboid (``--metadata-yaml``): max extent → max(cuboids_dims),
   applied about world origin after pivot.
5. **Final Y raise** (optional): min of ``y - σ·σ_y`` over **export-visible** splats only, with each
   splat's world σ_y **capped** vs. the visible Y extent (stabilizes ymin). Default ``dy = ymin - y_target``
   and ``y += dy`` (matches AH ``y + ty`` style for viewer pivot). The ground pivot skips **Y** when this
   runs (use ``--no-final-y-snap`` for AH ``y += ty`` in the pivot only). ``--invert-final-y-shift`` uses
   ``dy = y_target - ymin`` instead.

Writes processed PLY as ``gaussians.ply`` in the same directory as each source asset PLY (no tree copy).

Single file (``out`` = ``<dir-of-in>/gaussians.ply``):

  PYTHONPATH=models/tokengs python scripts/gaussian_post_processing.py path/to/asset.ply

Harvest folder (each ``<uuid>/<uuid>.ply`` → ``<uuid>/gaussians.ply``):

  PYTHONPATH=models/tokengs python scripts/gaussian_post_processing.py /path/to/harvested_084 --flip-axis x --also-flip-axis y --metadata-yaml /path/to/metadata.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import OrderedDict

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_TOKENGS_PKG = os.path.join(_REPO_ROOT, "models", "tokengs")
if _TOKENGS_PKG not in sys.path:
    sys.path.insert(0, _TOKENGS_PKG)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from scale_gaussian_ply import _gaussian_ply_extents, scale_gaussian_ply_vertex
from tokengs.ply_io import read_ply, write_ply

_PLY_PIVOT_Y_EXTENT_SIGMA = 2.0
_PLY_EXPORT_OPACITY_MIN = 0.005
# Cap world σ_y per splat as a fraction of visible Y AABB when computing final ground ymin.
# Uncapped min(y - σ·σ_y) can be dominated by a few elongated roof splats (wrong ymin → bad dy).
_FINAL_Y_HALF_Y_CAP_FRAC_OF_VISIBLE_Y_EXTENT = 0.4


def _rotation_180(axis: str) -> np.ndarray:
    """Return R (3, 3) with R @ R.T = I, det=+1, rotation by pi radians about world X, Y, or Z."""
    a = axis.lower()
    if a == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    if a == "y":
        return np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    if a == "z":
        return np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"axis must be x, y, or z, got {axis!r}")


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """q shape (4,) w,x,y,z -> (3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    m00 = ww + xx - yy - zz
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    m10 = 2.0 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2.0 * (yz - wx)
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = ww - xx - yy + zz
    return np.array([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]], dtype=np.float64)


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Single 3x3 rotation matrix -> (4,) wxyz."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def apply_flip_rigid(
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
    R_fix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    pos (N,3), quat (N,4) wxyz. Gaussian world rotation R_i; new R_i' = R_fix @ R_i, pos' = row @ R_fix.T.
    """
    n = pos.shape[0]
    pos_new = (R_fix @ pos.T).T.astype(np.float64)
    q_out = np.zeros_like(quat_wxyz, dtype=np.float64)
    for i in range(n):
        R_i = _quat_wxyz_to_rotmat(quat_wxyz[i])
        R_n = R_fix @ R_i
        q_out[i] = _rotmat_to_quat_wxyz(R_n)
    return pos_new, q_out


def apply_ground_plane_pivot(
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
    log_scales: np.ndarray,
    opacity_logit: np.ndarray,
    pivot_sigma: float,
    *,
    pivot_xz: str,
    apply_y_translation: bool,
) -> np.ndarray:
    """
    Return translated pos (N,3); quat and scales unchanged.

    When ``apply_y_translation`` is True, apply AH-style ``y += ty`` from visible splats. When False,
    only recenter X/Z so vertical alignment is left to the final world-Y snap (avoids double lift).
    """
    op_act = _sigmoid(opacity_logit.squeeze())
    vis = op_act >= _PLY_EXPORT_OPACITY_MIN
    if not np.any(vis):
        vis = np.ones_like(vis, dtype=bool)

    p = pos[vis]
    q = quat_wxyz[vis]
    ls = log_scales[vis]
    lin_s = np.exp(np.clip(ls, -20.0, 20.0))

    ty = 0.0
    if apply_y_translation:
        y_lo = []
        for i in range(p.shape[0]):
            R = _quat_wxyz_to_rotmat(q[i])
            s2 = lin_s[i] * lin_s[i]
            S2 = np.diag(s2)
            cov = R @ S2 @ R.T
            half_y = math.sqrt(max(0.0, cov[1, 1]))
            y_lo.append(p[i, 1] - pivot_sigma * half_y)
        y_lo = np.array(y_lo, dtype=np.float64)
        ty = float(y_lo.min())
    xmin, xmax = float(p[:, 0].min()), float(p[:, 0].max())
    zmin, zmax = float(p[:, 2].min()), float(p[:, 2].max())
    extent_x = xmax - xmin
    extent_z = zmax - zmin
    mode = pivot_xz.lower()
    if mode == "center":
        tx = 0.5 * (xmin + xmax)
        tz = 0.5 * (zmin + zmax)
    elif mode in ("rear", "front"):
        t_long = xmin if mode == "rear" else xmax
        t_long_z = zmin if mode == "rear" else zmax
        if extent_x >= extent_z:
            tx = t_long
            tz = 0.5 * (zmin + zmax)
        else:
            tx = 0.5 * (xmin + xmax)
            tz = t_long_z
    else:
        raise ValueError(f"pivot_xz must be center, rear, or front, got {pivot_xz!r}")

    out = pos.astype(np.float64).copy()
    out[:, 0] -= tx
    if apply_y_translation:
        out[:, 1] += ty
    out[:, 2] -= tz
    return out.astype(np.float32)


def _raise_lowest_world_y_extent(
    vertex: OrderedDict,
    *,
    pivot_sigma: float,
    y_target: float,
    invert_final_y_shift: bool,
) -> None:
    """
    Shift all splats in Y using the minimum lower world-Y bound ymin (AH-style envelope).

    Uses the same opacity mask as PLY export (≥ ``_PLY_EXPORT_OPACITY_MIN``) and caps each splat's
    world σ_y when building y_lo so a few huge roof ellipsoids cannot set ymin (or bogus highs from
    junk splats skewing the aggregate).

    Default ``dy = ymin - y_target`` and ``y += dy``. If ``invert_final_y_shift``, use
    ``dy = y_target - ymin`` instead when the mesh hangs the wrong way after flips.
    """
    pos = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float64),
            np.asarray(vertex["y"], dtype=np.float64),
            np.asarray(vertex["z"], dtype=np.float64),
        ],
        axis=1,
    )
    op = np.asarray(vertex["opacity"], np.float64)
    if op.ndim == 1:
        op = op[:, np.newaxis]
    op_act = _sigmoid(op.squeeze())
    vis = op_act >= _PLY_EXPORT_OPACITY_MIN
    if not np.any(vis):
        vis = np.ones_like(vis, dtype=bool)

    rot_names = sorted(k for k in vertex if k.startswith("rot_"))
    quat = np.stack([np.asarray(vertex[k], np.float64) for k in rot_names], axis=1)
    scale_names = sorted(k for k in vertex if k.startswith("scale_"))
    log_scales = np.stack([np.asarray(vertex[k], np.float64) for k in scale_names], axis=1)
    lin_s = np.exp(np.clip(log_scales, -20.0, 20.0))
    n = pos.shape[0]
    bbox_y = float(pos[vis, 1].max() - pos[vis, 1].min())
    half_y_cap = max(bbox_y * _FINAL_Y_HALF_Y_CAP_FRAC_OF_VISIBLE_Y_EXTENT, 1e-8)
    y_lo = np.empty(n, dtype=np.float64)
    for i in range(n):
        R = _quat_wxyz_to_rotmat(quat[i])
        s2 = lin_s[i] * lin_s[i]
        S2 = np.diag(s2)
        cov = R @ S2 @ R.T
        half_y = math.sqrt(max(0.0, cov[1, 1]))
        half_y = min(half_y, half_y_cap)
        y_lo[i] = pos[i, 1] - pivot_sigma * half_y
    ymin = float(np.min(y_lo[vis]))
    if invert_final_y_shift:
        dy = float(y_target) - ymin
    else:
        dy = ymin - float(y_target)
    if not math.isfinite(dy):
        return
    pos[:, 1] += dy
    vertex["x"] = pos[:, 0].astype(np.float32)
    vertex["y"] = pos[:, 1].astype(np.float32)
    vertex["z"] = pos[:, 2].astype(np.float32)
    mode = "inverted (y_target - ymin)" if invert_final_y_shift else "default (ymin - y_target)"
    print(
        f"  final Y [{mode}]: ymin={ymin:.6g} (visible, σ_y cap={half_y_cap:.6g}), all y += {dy:+.6g}"
    )


def process_ply(
    vertex: OrderedDict,
    *,
    flip_axis: str | None,
    also_flip_axis: str | None,
    apply_pivot: bool,
    pivot_sigma: float,
    pivot_xz: str,
    pivot_apply_y: bool,
) -> OrderedDict:
    pos = np.stack(
        [np.asarray(vertex["x"], np.float64), np.asarray(vertex["y"], np.float64), np.asarray(vertex["z"], np.float64)],
        axis=1,
    )
    rot_names = sorted(k for k in vertex if k.startswith("rot_"))
    if len(rot_names) != 4:
        raise ValueError(f"Expected 4 rot_* properties, got {rot_names}")
    quat = np.stack([np.asarray(vertex[k], np.float64) for k in rot_names], axis=1)

    scale_names = sorted(k for k in vertex if k.startswith("scale_"))
    log_scales = np.stack([np.asarray(vertex[k], np.float64) for k in scale_names], axis=1)

    op = np.asarray(vertex["opacity"], np.float64)
    if op.ndim == 1:
        op = op[:, np.newaxis]

    if flip_axis is not None:
        R_fix = _rotation_180(flip_axis)
        pos, quat = apply_flip_rigid(pos, quat, R_fix)
    if also_flip_axis is not None:
        R2 = _rotation_180(also_flip_axis)
        pos, quat = apply_flip_rigid(pos, quat, R2)

    if apply_pivot:
        pos = apply_ground_plane_pivot(
            pos,
            quat,
            log_scales,
            op,
            pivot_sigma,
            pivot_xz=pivot_xz,
            apply_y_translation=pivot_apply_y,
        )

    out = OrderedDict()
    for key in vertex.keys():
        if key == "x":
            out["x"] = pos[:, 0].astype(np.float32)
        elif key == "y":
            out["y"] = pos[:, 1].astype(np.float32)
        elif key == "z":
            out["z"] = pos[:, 2].astype(np.float32)
        elif key.startswith("rot_"):
            idx = int(key.rsplit("_", 1)[-1])
            out[key] = quat[:, idx].astype(np.float32)
        else:
            out[key] = np.asarray(vertex[key], dtype=np.float32)

    return out


def _is_harvest_asset_ply(dirpath: str, filename: str) -> bool:
    """NRE layout: <track_id>/<track_id>.ply."""
    if not filename.endswith(".ply"):
        return False
    if filename == "gaussians.ply":
        return False
    parent = os.path.basename(dirpath)
    stem = filename[:-4]
    return stem == parent


def _load_metadata_target_extents(metadata_yaml: str) -> dict[str, float]:
    """Map track_id -> max(cuboids_dims) in meters (NRE metadata.yaml)."""
    try:
        import yaml
    except ImportError as e:
        print("error: PyYAML is required for --metadata-yaml (pip install pyyaml).", file=sys.stderr)
        raise SystemExit(1) from e
    with open(metadata_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assets = data.get("assets") if isinstance(data, dict) else None
    if not isinstance(assets, dict):
        return {}
    out: dict[str, float] = {}
    for track_id, entry in assets.items():
        if not isinstance(entry, dict):
            continue
        cd = entry.get("cuboids_dims")
        if not isinstance(cd, (list, tuple)) or len(cd) < 3:
            continue
        try:
            out[str(track_id)] = max(float(cd[0]), float(cd[1]), float(cd[2]))
        except (TypeError, ValueError):
            continue
    return out


def _write_processed_ply(
    input_ply: str,
    output_ply: str,
    *,
    flip_axis: str | None,
    also_flip_axis: str | None,
    apply_pivot: bool,
    pivot_sigma: float,
    pivot_xz: str,
    target_max_extent: float | None,
    pivot_y_lift: float,
    skip_final_y_snap: bool,
    invert_final_y_shift: bool,
) -> int:
    """Read; flip, flip, pivot; optional scale; final world-Y bound snap; write."""
    raw = read_ply(input_ply)
    vertex = OrderedDict(raw) if not isinstance(raw, OrderedDict) else raw
    out_vertex = process_ply(
        vertex,
        flip_axis=flip_axis,
        also_flip_axis=also_flip_axis,
        apply_pivot=apply_pivot,
        pivot_sigma=pivot_sigma,
        pivot_xz=pivot_xz,
        pivot_apply_y=skip_final_y_snap,
    )
    if target_max_extent is not None:
        _, max_extent = _gaussian_ply_extents(out_vertex)
        if max_extent <= 0 or not math.isfinite(max_extent):
            print(f"warning: skip scale for {input_ply} (invalid extent).", file=sys.stderr)
        else:
            s = target_max_extent / max_extent
            scale_gaussian_ply_vertex(out_vertex, s, anchor="origin")
            print(f"  scale s={s:.6g} (target max extent={target_max_extent:.6g}, anchor=origin, after pivot)")
    if not skip_final_y_snap:
        _raise_lowest_world_y_extent(
            out_vertex,
            pivot_sigma=pivot_sigma,
            y_target=pivot_y_lift,
            invert_final_y_shift=invert_final_y_shift,
        )
    out_dir = os.path.dirname(os.path.abspath(output_ply))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        write_ply(output_ply, len(out_vertex["x"]), out_vertex)
    except PermissionError as e:
        print(
            "error: cannot write output (permission denied). "
            "If this tree was produced by Docker as root, run on the host: "
            "sudo chown -R \"$USER:$USER\" <harvest_dir>. "
            "For the next harvest, use scripts/run_ah_docker.sh (runs as your uid:gid by default).\n"
            f"{e}",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    return int(len(out_vertex["x"]))


def _run_batch(input_dir: str, args: argparse.Namespace) -> None:
    root_dir = os.path.abspath(input_dir)
    if not os.path.isdir(root_dir):
        print(f"error: not a directory: {root_dir}", file=sys.stderr)
        raise SystemExit(2)
    flip = None if args.no_flip else args.flip_axis
    also_flip = None if args.no_flip else args.also_flip_axis
    apply_pivot = not args.no_pivot
    meta_targets: dict[str, float] = {}
    if args.metadata_yaml:
        my = os.path.abspath(args.metadata_yaml)
        if not os.path.isfile(my):
            print(f"error: metadata file not found: {my}", file=sys.stderr)
            raise SystemExit(2)
        meta_targets = _load_metadata_target_extents(my)
        print(f"Loaded {len(meta_targets)} cuboid target extents from {my}")
    count = 0
    total_splats = 0
    for root, _, files in os.walk(root_dir):
        for f in files:
            if not _is_harvest_asset_ply(root, f):
                continue
            in_ply = os.path.join(root, f)
            out_ply = os.path.join(root, "gaussians.ply")
            track_id = os.path.basename(root)
            target_max = meta_targets.get(track_id) if meta_targets else None
            if meta_targets and target_max is None:
                print(f"warning: no cuboids_dims for track {track_id}, skipping scale.", file=sys.stderr)
            n = _write_processed_ply(
                in_ply,
                out_ply,
                flip_axis=flip,
                also_flip_axis=also_flip,
                apply_pivot=apply_pivot,
                pivot_sigma=args.pivot_sigma,
                pivot_xz=args.pivot_xz,
                target_max_extent=target_max,
                pivot_y_lift=args.pivot_y_lift,
                skip_final_y_snap=args.no_final_y_snap,
                invert_final_y_shift=args.invert_final_y_shift,
            )
            print(f"Wrote {out_ply} ({n} splats) from {in_ply}")
            total_splats += n
            count += 1
    print(f"Batch done: {count} asset(s), {total_splats} total splats under {root_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix orientation and pivot for Gaussian PLY.")
    parser.add_argument(
        "input_path",
        help="Input .ply file or harvest directory; output is always <same-dir>/gaussians.ply.",
    )
    parser.add_argument(
        "--flip-axis",
        choices=("x", "y", "z"),
        default="x",
        help="First 180° world rotation (default: x = roll / upside-down fix).",
    )
    parser.add_argument(
        "--also-flip-axis",
        choices=("x", "y", "z"),
        default=None,
        help="Second 180° rotation after --flip-axis (e.g. y for front-back in Y-up).",
    )
    parser.add_argument(
        "--metadata-yaml",
        default="",
        metavar="PATH",
        help="NRE harvest metadata.yaml; scale each asset so max AABB extent matches max(cuboids_dims).",
    )
    parser.add_argument("--no-flip", action="store_true", help="Skip all 180° rotations.")
    parser.add_argument("--no-pivot", action="store_true", help="Skip ground-plane pivot.")
    parser.add_argument(
        "--pivot-sigma",
        type=float,
        default=_PLY_PIVOT_Y_EXTENT_SIGMA,
        help="Vertical envelope factor for pivot (default matches AH save_ply).",
    )
    parser.add_argument(
        "--pivot-xz",
        choices=("center", "rear", "front"),
        default="center",
        help=(
            "Horizontal origin: center = midpoint on X and Z (bottom-center footprint). "
            "rear/front = min or max on the longer horizontal span only; after flips that may be bumper "
            "not vehicle rear—use center unless you want axle-style placement."
        ),
    )
    parser.add_argument(
        "--pivot-y-lift",
        type=float,
        default=0.05,
        metavar="M",
        help="Final stage: target world Y (meters) for the lowest Gaussian lower bound (see module "
        "doc); default 0.05 clears typical grids slightly.",
    )
    parser.add_argument(
        "--no-final-y-snap",
        action="store_true",
        help="Skip the final lowest-point Y raise.",
    )
    parser.add_argument(
        "--invert-final-y-shift",
        action="store_true",
        help="Use dy = y_target - ymin instead of default dy = ymin - y_target.",
    )
    args = parser.parse_args()
    inp = os.path.abspath(args.input_path)

    if os.path.isdir(inp):
        _run_batch(inp, args)
        return

    if not os.path.isfile(inp):
        print(f"error: not a file or directory: {inp}", file=sys.stderr)
        raise SystemExit(2)

    out_ply = os.path.join(os.path.dirname(inp), "gaussians.ply")
    flip = None if args.no_flip else args.flip_axis
    also_flip = None if args.no_flip else args.also_flip_axis
    track_id = os.path.basename(os.path.dirname(inp))
    target_max: float | None = None
    if args.metadata_yaml:
        my = os.path.abspath(args.metadata_yaml)
        if not os.path.isfile(my):
            print(f"error: metadata file not found: {my}", file=sys.stderr)
            raise SystemExit(2)
        meta_targets = _load_metadata_target_extents(my)
        target_max = meta_targets.get(track_id)
        if target_max is None:
            print(f"warning: no cuboids_dims for track {track_id}, skipping scale.", file=sys.stderr)
    n = _write_processed_ply(
        inp,
        out_ply,
        flip_axis=flip,
        also_flip_axis=also_flip,
        apply_pivot=not args.no_pivot,
        pivot_sigma=args.pivot_sigma,
        pivot_xz=args.pivot_xz,
        target_max_extent=target_max,
        pivot_y_lift=args.pivot_y_lift,
        skip_final_y_snap=args.no_final_y_snap,
        invert_final_y_shift=args.invert_final_y_shift,
    )
    print(
        f"Wrote {out_ply} ({n} splats), flip={flip}, also_flip={also_flip}, "
        f"pivot={not args.no_pivot}, pivot_xz={args.pivot_xz}"
    )


if __name__ == "__main__":
    main()
