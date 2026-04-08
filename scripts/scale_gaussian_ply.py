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
Measure axis-aligned size of an Asset Harvester / TokenGS Gaussian PLY and rescale uniformly.

Positions (x,y,z) are multiplied by s. In compatible PLY, scale_* are log(linear scale);
those get log(s) added so exp(scale) scales like the means.

Example from repo root:

  PYTHONPATH=models/tokengs python scripts/scale_gaussian_ply.py in.ply out.ply --target-max 4.5
  PYTHONPATH=models/tokengs python scripts/scale_gaussian_ply.py in.ply --measure-only
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

from tokengs.ply_io import read_ply, write_ply


def _gaussian_ply_extents(vertex: dict) -> tuple[np.ndarray, float]:
    """Return per-axis AABB size (max-min) and the largest axis extent."""
    x = np.asarray(vertex["x"], dtype=np.float64)
    y = np.asarray(vertex["y"], dtype=np.float64)
    z = np.asarray(vertex["z"], dtype=np.float64)
    xyz = np.stack([x, y, z], axis=1)
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    extent = hi - lo
    return extent, float(np.max(extent))


def scale_gaussian_ply_vertex(
    vertex: dict,
    scale: float,
    about_bbox_center: bool = False,
    *,
    anchor: str | None = None,
) -> None:
    """
    Mutate vertex dict in place: positions and log-scales (compatible PLY).

    Scaling anchor (how the mean positions are fixed under uniform scale):

    - ``origin`` (default): multiply x,y,z by scale (``about_bbox_center=False``).
    - ``bbox_center``: fixed point is the 3D AABB center (``about_bbox_center=True``).
    - ``bottom_center``: fixed point is (mid X, min Y, mid Z) of splat centers — keeps the underside
      from drifting before a ground pivot (preferred before flip/pivot in gaussian_post_processing).

    If ``anchor`` is set, it overrides ``about_bbox_center``.
    """
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError(f"scale must be positive finite, got {scale}")

    x = np.asarray(vertex["x"], dtype=np.float64)
    y = np.asarray(vertex["y"], dtype=np.float64)
    z = np.asarray(vertex["z"], dtype=np.float64)

    if anchor is None:
        anchor = "bbox_center" if about_bbox_center else "origin"
    if anchor not in ("origin", "bbox_center", "bottom_center"):
        raise ValueError(f"anchor must be origin, bbox_center, or bottom_center, got {anchor!r}")

    if anchor == "origin":
        x = scale * x
        y = scale * y
        z = scale * z
    elif anchor == "bbox_center":
        cx = 0.5 * (x.min() + x.max())
        cy = 0.5 * (y.min() + y.max())
        cz = 0.5 * (z.min() + z.max())
        x = cx + scale * (x - cx)
        y = cy + scale * (y - cy)
        z = cz + scale * (z - cz)
    else:
        cx = 0.5 * (x.min() + x.max())
        cy = float(y.min())
        cz = 0.5 * (z.min() + z.max())
        x = cx + scale * (x - cx)
        y = cy + scale * (y - cy)
        z = cz + scale * (z - cz)
    vertex["x"] = x.astype(np.float32)
    vertex["y"] = y.astype(np.float32)
    vertex["z"] = z.astype(np.float32)

    log_s = math.log(scale)
    for key in sorted(k for k in vertex if k.startswith("scale_")):
        arr = np.asarray(vertex[key], dtype=np.float64) + log_s
        vertex[key] = arr.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uniform scale for TokenGS-compatible Gaussian PLY (means + log scale_*).",
    )
    parser.add_argument("input_ply", help="Input .ply path.")
    parser.add_argument(
        "output_ply",
        nargs="?",
        default="",
        help="Output .ply path (omit with --measure-only).",
    )
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument(
        "--target-max",
        type=float,
        default=None,
        metavar="M",
        help="Set largest axis-aligned extent of splat centers to this length (same units as the file).",
    )
    g.add_argument(
        "--scale",
        type=float,
        default=None,
        metavar="S",
        help="Multiply all positions by S and linear Gaussian scales by S (log scale += log(S)).",
    )
    g_anchor = parser.add_mutually_exclusive_group()
    g_anchor.add_argument(
        "--about-bbox-center",
        action="store_true",
        help="Scale about the 3D AABB center (can shift the underside vertically).",
    )
    g_anchor.add_argument(
        "--about-bottom-center",
        action="store_true",
        help="Scale about (mid X, min Y, mid Z) — keeps ground contact through scale.",
    )
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help="Print extents and exit (no output file written).",
    )
    args = parser.parse_args()
    if not args.measure_only and args.target_max is None and args.scale is None:
        print("error: specify --target-max or --scale (or use --measure-only).", file=sys.stderr)
        raise SystemExit(2)

    vertex = read_ply(args.input_ply)
    if not isinstance(vertex, OrderedDict):
        vertex = OrderedDict(vertex)

    extent, max_extent = _gaussian_ply_extents(vertex)
    print(
        f"Axis-aligned extent of splat centers (min corner to max corner): "
        f"x={extent[0]:.6g}, y={extent[1]:.6g}, z={extent[2]:.6g} (max={max_extent:.6g})"
    )

    if args.measure_only:
        return

    if not args.output_ply:
        print("error: output_ply is required unless using --measure-only.", file=sys.stderr)
        raise SystemExit(2)

    if args.scale is not None:
        s = args.scale
    else:
        if max_extent <= 0:
            raise ValueError("Cannot scale: zero or invalid extent.")
        s = args.target_max / max_extent

    print(f"Applying uniform scale factor s={s:.6g}")
    if args.about_bottom_center:
        scale_gaussian_ply_vertex(vertex, s, anchor="bottom_center")
    elif args.about_bbox_center:
        scale_gaussian_ply_vertex(vertex, s, anchor="bbox_center")
    else:
        scale_gaussian_ply_vertex(vertex, s, anchor="origin")

    out = OrderedDict((k, vertex[k]) for k in vertex.keys())
    n = len(out["x"])
    write_ply(args.output_ply, n, out)
    print(f"Wrote {args.output_ply} ({n} splats)")


if __name__ == "__main__":
    main()
