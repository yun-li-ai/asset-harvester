# Single-Image Asset Harvesting

This guide documents how to turn a **single object image** (e.g. a car photo from the web) into a 3D Gaussian splat asset using Asset Harvester — without NCore driving-log data.

This is the workflow used for the `data_samples/pandaset_misc/` examples.

## Prerequisites

Complete the [setup and checkpoint download](../README.md#user-guide) steps from the main README. You need these checkpoints:

| Checkpoint | Path |
|------------|------|
| Segmentation | `checkpoints/AH_object_seg_jit.pt` |
| Multiview diffusion | `checkpoints/AH_multiview_diffusion.safetensors` |
| Camera estimator | `checkpoints/AH_camera_estimator.safetensors` |
| Gaussian lifting (TokenGS) | `checkpoints/AH_tokengs_lifting.safetensors` |

## Folder Layout

Put each object in its own subfolder under a root directory. One source image per folder.

```
data_samples/pandaset_misc/
├── hyundai_elantra_2020_black/
│   └── 1.jpeg                          # raw source image (any name)
├── red_toyota_prius/
│   └── 001.jpg
└── toyota_camery/
    └── 2007_toyota_camry_angularfront.jpg
```

After the full pipeline, each folder should contain:

```
hyundai_elantra_2020_black/
├── 1.jpeg          # original source
├── frame.jpeg      # 512×512 prepared input
└── mask.png        # segmentation mask
```

`run_inference.py` pairs `frame.jpeg` with `mask.png` automatically (`frame` → `mask` naming convention).

## Step 1: Prepare 512×512 Frames

Source images are often wide landscape crops (e.g. 311×162). The pipeline expects square **512×512** inputs.

From the repo root:

```bash
./prepare_frame.sh
```

This walks every subfolder under `data_samples/pandaset_misc/`, picks the first source image (skipping `frame.jpeg` and `mask.png`), and writes `frame.jpeg` in each folder.

| Env var | Default | Description |
|---------|---------|-------------|
| `INPUT_ROOT` | `data_samples/pandaset_misc` | Root folder containing one subfolder per object |
| `MODE` | `pad` | How to make the image square (see below) |
| `OUTPUT_NAME` | `frame.jpeg` | Output filename |

**Modes:**

- **`pad`** (default) — pad with white borders to a square, then resize to 512×512. Best for car-on-white photos where the subject spans most of the width.
- **`crop`** — center square crop, then resize. Cuts off left/right edges.
- **`stretch`** — direct resize to 512×512. Distorts aspect ratio.

Examples:

```bash
# Process a different root folder
INPUT_ROOT=data_samples/my_cars ./prepare_frame.sh

# Center-crop instead of pad
MODE=crop ./prepare_frame.sh

# Single image via Python directly
python3 utils/prepare_frame_image.py path/to/image.jpg --mode pad
```

## Step 2: Generate Segmentation Masks

Run Mask2Former segmentation on every `frame.jpeg` and write `mask.png` alongside it:

```bash
./run_mask.sh
```

`run_mask.sh` is configured to search `data_samples/pandaset_misc/` recursively. Edit `IMAGE_ROOT` in the script to point at a different folder.

## Step 3: Run Inference

Generate multiview images, lift to 3D Gaussians, and export the final asset:

```bash
./run_command_single.sh
```

Or run `run_inference.py` directly:

```bash
python3 run_inference.py \
    --diffusion_checkpoint checkpoints/AH_multiview_diffusion.safetensors \
    --ahc_checkpoint checkpoints/AH_camera_estimator.safetensors \
    --lifting_checkpoint checkpoints/AH_tokengs_lifting.safetensors \
    --image_dir data_samples/pandaset_misc/ \
    --output_dir outputs/pandaset_misc_v1
```

| Argument | Description |
|----------|-------------|
| `--image_dir` | Root folder with one subfolder per object (`frame.jpeg` + `mask.png` in each) |
| `--output_dir` | Where to write results |
| `--ahc_checkpoint` | Required for `image_dir` mode (estimates camera pose from a single view) |

## Outputs

Per object, under `outputs/pandaset_misc_v1/<folder_name>/`:

| File / folder | Description |
|---------------|-------------|
| `gaussians.ply` | 3D Gaussian splat asset |
| `multiview/` | Generated novel-view images |
| `3d_lifted/` | TokenGS-rendered views |
| `multiview.mp4` | Video of generated views |
| `3d_lifted.mp4` | Video of lifted 3D render |
| `input/` | Copy of input frame and mask |

Example:

```
outputs/pandaset_misc_v1/
├── hyundai_elantra_2020_black/
│   ├── gaussians.ply
│   ├── multiview.mp4
│   └── 3d_lifted.mp4
├── red_toyota_prius/
│   └── ...
└── toyota_camery/
    └── ...
```

## Step 4: View in SuperSplat

Open [superspl.at/editor](https://superspl.at/editor) and load `gaussians.ply`.

**Important:** use **drag-and-drop** or **File → Import**. Do **not** use **File → Open** — that path can misidentify `.ply` files as zip archives and show:

> `End of central directory not found - invalid zip file`

The exported `gaussians.ply` is a valid 3D Gaussian Splat file. If import still fails, convert to compressed PLY:

```bash
npm install -g @playcanvas/splat-transform
splat-transform \
  outputs/pandaset_misc_v1/hyundai_elantra_2020_black/gaussians.ply \
  outputs/pandaset_misc_v1/hyundai_elantra_2020_black/gaussians.compressed.ply
```

Then drag-and-drop the `.compressed.ply` file.

## Quick Reference (Full Pipeline)

```bash
# 1. Add source images: data_samples/pandaset_misc/<name>/<image>.jpg

# 2. Prepare 512×512 frames
./prepare_frame.sh

# 3. Generate masks
./run_mask.sh

# 4. Run inference
./run_command_single.sh

# 5. View gaussians.ply in SuperSplat (drag-and-drop)
```

## Adding More Objects

1. Create a new subfolder under `data_samples/pandaset_misc/`.
2. Drop a source image into it (`.jpg`, `.jpeg`, or `.png`).
3. Re-run steps 1–4 above. Existing folders are re-processed too; delete stale `frame.jpeg` / `mask.png` first if you only changed the source image.

## Related Docs

- [Full end-to-end example (NCore driving logs)](end_to_end_example.md) — the standard pipeline from NCore V4 clips.
- [Main README](../README.md) — setup, checkpoints, and Docker usage.
