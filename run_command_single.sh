export YOUR_IMAGE_ROOT=data_samples/pandaset_misc/
export CHECKPOINT_MV=checkpoints/AH_multiview_diffusion.safetensors
export CHECKPOINT_GS=checkpoints/AH_tokengs_lifting.safetensors
export CHECKPOINT_CAM=checkpoints/AH_camera_estimator.safetensors
export OUTPUT_DIR=outputs/pandaset_misc_v1
python3 run_inference.py \
    --diffusion_checkpoint "${CHECKPOINT_MV}" \
    --ahc_checkpoint "${CHECKPOINT_CAM}" \
    --image_dir "${YOUR_IMAGE_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --lifting_checkpoint  "${CHECKPOINT_GS}"