export CHECKPOINT_SEG=checkpoints/AH_object_seg_jit.pt
export IMAGE_ROOT=data_samples/pandaset_misc/
python utils/image_segment.py \
  --checkpoint $CHECKPOINT_SEG \
  --image_folder $IMAGE_ROOT \
  --frame_name frame.jpeg \
  --mask_name mask.png