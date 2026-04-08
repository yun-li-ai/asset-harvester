#!/usr/bin/env sh
# Run from asset-harvester repo root. Writes gaussians.ply next to each source PLY (same tree as input).
# Roll fix (X), front-back (Y), XZ pivot + cuboid scale, then final Y snap (default dy = ymin - y_target; no double Y).
# Other convention: add --invert-final-y-shift (dy = y_target - ymin).
# Override: --pivot-y-lift 0  |  disable: --no-final-y-snap
#
# If harvest was created by Docker as root, fix ownership once on the host, then re-run this script:
#   sudo chown -R "$USER:$USER" output-tools/harvested_<SEQ>
# Or re-harvest with scripts/run_ah_docker.sh (uses --user uid:gid by default).
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_SEQUENCE="016"

HARVEST_IN="${HARVEST_IN:-$REPO_ROOT/output-tools/harvested_${DATASET_SEQUENCE}}"
META_YAML="${META_YAML:-$HARVEST_IN/metadata.yaml}"
if [ -f "$META_YAML" ]; then
  PYTHONPATH="$REPO_ROOT/models/tokengs" python3 scripts/gaussian_post_processing.py "$HARVEST_IN" \
    --flip-axis x --also-flip-axis y --pivot-xz center --metadata-yaml "$META_YAML"
else
  echo "warning: metadata not found at $META_YAML; skipping cuboid scaling." >&2
  PYTHONPATH="$REPO_ROOT/models/tokengs" python3 scripts/gaussian_post_processing.py "$HARVEST_IN" \
    --flip-axis x --also-flip-axis y --pivot-xz center
fi
echo "Wrote gaussians.ply under each asset folder in: $HARVEST_IN"

# Single asset (uncomment):
# UUID="0d3641ac-5709-4c1b-b3bf-d17dc89486cb"
# PYTHONPATH="$REPO_ROOT/models/tokengs" python3 scripts/gaussian_post_processing.py \
#   "$HARVEST_IN/${UUID}/${UUID}.ply" --flip-axis x --also-flip-axis y --pivot-xz center --metadata-yaml "$META_YAML"
