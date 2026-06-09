#!/usr/bin/env bash
set -euo pipefail

INPUT_ROOT="${INPUT_ROOT:-data_samples/pandaset_misc}"
MODE="${MODE:-pad}"  # pad | crop | stretch
OUTPUT_NAME="${OUTPUT_NAME:-frame.jpeg}"

shopt -s nullglob

for sample_dir in "${INPUT_ROOT}"/*/; do
  prepared=0
  for image_path in \
    "${sample_dir}"*.jpg \
    "${sample_dir}"*.jpeg \
    "${sample_dir}"*.png \
    "${sample_dir}"*.webp \
    "${sample_dir}"*.bmp; do
    image_name="$(basename "${image_path}")"
    if [[ "${image_name}" == "${OUTPUT_NAME}" || "${image_name}" == "mask.png" ]]; then
      continue
    fi

    echo "Preparing ${image_path}"
    python3 utils/prepare_frame_image.py \
      "${image_path}" \
      --mode "${MODE}" \
      --output-name "${OUTPUT_NAME}"
    prepared=1
    break
  done

  if [[ "${prepared}" -eq 0 ]]; then
    echo "No source image found in ${sample_dir}" >&2
  fi
done
