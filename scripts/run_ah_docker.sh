DATASET_SEQUENCE="016"
# Asset-harvester repo root (this file lives in scripts/).
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Write harvest output as your host user so post-processing can add gaussians.ply without sudo.
# Set AH_DOCKER_AS_ROOT=1 to omit (container default root; you will need chown on the host).
DOCKER_USER_ARG=
if [ "${AH_DOCKER_AS_ROOT:-0}" != "1" ]; then
  DOCKER_USER_ARG="--user $(id -u):$(id -g)"
fi

docker run -it --rm \
  $DOCKER_USER_ARG \
  --gpus=all \
  -e NGC_API_KEY=${NGC_API_KEY} \
  --volume ${ROOT_DIR}/pandaset_ncore_logs:/workdir/dataset \
  --volume ${ROOT_DIR}/output-tools:/workdir/output \
  --volume ${ROOT_DIR}/asset_harvester_v26.02:/cache \
  nvcr.io/nvidia/nre/nre-tools:26.02 \
  asset-harvester \
  --shard-file-pattern="/workdir/dataset/${DATASET_SEQUENCE}/${DATASET_SEQUENCE}.zarr.itar" \
  --output-dir="/workdir/output/harvested_${DATASET_SEQUENCE}" \
  --cache-dir="/cache" \
  views_extractor.camera_ids=["back_camera","front_camera","front_left_camera","front_right_camera","left_camera","right_camera"] \
  views_extractor.crop_labels=["Car","Pedestrian","Truck","Bus","Bicycle","Motorcycle"] \
  gaussian_lift.gaussian_type="3dgs"