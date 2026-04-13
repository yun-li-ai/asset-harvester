# Scripts quick start

Run these steps from the **asset-harvester repo root** (parent of `scripts/`).

## 1. Download dependency models (NGC)

Requires the [NGC CLI](https://docs.nvidia.com/ngc/ngc-cli-overview/) and a valid `NGC_API_KEY`.

```bash
cd /path/to/asset-harvester
ngc registry model download-version "nvidia/nre/asset_harvester:26.02"
```

Example JSON response (paths vary by machine):

```json
{
  "download_end": "2026-04-07 15:25:36",
  "download_start": "2026-04-07 15:23:49",
  "download_time": "1m 47s",
  "files_downloaded": 3,
  "local_path": "/home/you/code/asset-harvester/asset_harvester_v26.02",
  "size_downloaded": "11.29 GB",
  "status": "Completed",
  "transfer_id": "asset_harvester[version=26.02]"
}
```

The cache directory must end up at **`./asset_harvester_v26.02`** next to the repo root (see `run_ah_docker.sh`).

## 2. Dataset layout

Downloade data from here https://drive.google.com/drive/folders/1wjecqxSw0uIDJFlBmzF-giI-7tv1t5iP?usp=sharing

Place **pandaset_ncore_logs** (or your shard source tree) under:

```text
./pandaset_ncore_logs/<SEQ>/<SEQ>.zarr.itar
```

Default sequence in `run_ah_docker.sh` is **`016`**; change `DATASET_SEQUENCE` in that script if you use another shard.

## 3. Run Asset Harvester (Docker)

```bash
export NGC_API_KEY=...   # if the image needs it
sh scripts/run_ah_docker.sh
```

By default the container runs as **`--user $(id -u):$(id -g)`** so files under `output-tools/` are writable on the host without `sudo`. That avoids permission errors when you run Gaussian post-processing.

If you need the container as root instead:

```bash
AH_DOCKER_AS_ROOT=1 sh scripts/run_ah_docker.sh
```

Then fix ownership once before post-processing, for example:

```bash
sudo chown -R "$USER:$USER" output-tools/harvested_016
```

Harvest output path (default): **`output-tools/harvested_<SEQ>/`**.

## 4. Gaussian PLY post-processing

From the repo root:

```bash
sh scripts/run_gaussian_post_processing.sh
```

This writes **`gaussians.ply`** next to each source asset PLY under the harvest directory. It uses **`metadata.yaml`** in that directory for cuboid scaling when present; otherwise it skips scaling and prints a warning.

Useful overrides (see script comments):

- **`HARVEST_IN`** — path to a harvest folder (default: `output-tools/harvested_<SEQ>`).
- **`META_YAML`** — explicit metadata path.
- Edit **`DATASET_SEQUENCE`** at the top of `run_gaussian_post_processing.sh` to match your harvest.

Single-asset and extra CLI flags are documented in **`gaussian_post_processing.py`** and in comments at the bottom of `run_gaussian_post_processing.sh`.
