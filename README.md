# Training-free object extraction and distractor removal for 3D Gaussian Splatting

Given a trained 3D Gaussian Splatting scene and a text prompt naming the object of interest (for example, "red roses."), this pipeline keeps the Gaussians that belong to that object and prunes the background and floating artefacts. It needs no retraining: the decision is made from 2D masks and the camera poses alone.

Evaluated on the 12 scenes of the [Nerfbusters](https://ethanweber.me/nerfbusters/) dataset.

## Method

1. **Train** an initial splat per scene with [OpenSplat](https://github.com/pierotofy/OpenSplat) (5,000 steps) from the COLMAP poses (`run_all.sh`).
2. **Mask** the object in every frame (`generate_masks_sam2.py`, `run_all_masks.sh`). Grounding DINO (tiny) finds the prompted object in the first frame, and SAM 2.1 (Hiera-L) video propagation tracks its mask through the capture. Each scene runs in a separate process, which keeps memory stable on Apple Silicon (MPS).
3. **Vote** (`01_extract_objects.py`). Each Gaussian's centre is projected into every view. It gains a vote when it lands inside the mask and loses one when it lands outside. A Gaussian is kept if its normalised vote exceeds a threshold γ.
4. **Evaluate** (`02_evaluate_pipeline.py`): file size, run time and peak memory of the voting step, and PSNR, SSIM and LPIPS inside the object mask (see below). `02_evaluate_pipeline_tuned.py` is the same script with a per-scene γ.

γ trades the two failure modes. A low γ keeps floaters and background; a high γ removes thin or occluded parts of the object. No single value worked well for every scene, so per-scene values (from −0.2 to 0.6) are set in `SCENE_GAMMAS` in `02_evaluate_pipeline_tuned.py`, with `tune.sh` for quick trials.

## Results

| Scene | Size before → after (MB) | Reduction | Voting time (s) | Peak RAM (MB) | PSNR before → after | SSIM before → after | LPIPS before → after |
|---|---|---|---|---|---|---|---|
| aloe | 187.2 → 14.1 | 92.5% | 17.2 | 341 | 17.62 → 22.01 | 0.871 → 0.904 | 0.161 → 0.142 |
| art | 143.6 → 45.8 | 68.1% | 10.6 | 308 | 13.73 → 19.23 | 0.760 → 0.800 | 0.276 → 0.235 |
| car | 163.3 → 14.8 | 90.9% | 13.3 | 296 | 18.12 → 19.12 | 0.817 → 0.833 | 0.276 → 0.266 |
| century | 70.9 → 38.6 | 45.6% | 7.7 | 186 | 14.84 → 14.73 | 0.620 → 0.618 | 0.355 → 0.361 |
| flowers | 63.0 → 6.2 | 90.2% | 4.2 | 146 | 17.15 → 18.20 | 0.729 → 0.727 | 0.242 → 0.238 |
| garbage | 99.2 → 30.1 | 69.7% | 7.3 | 230 | 21.29 → 21.98 | 0.862 → 0.869 | 0.195 → 0.196 |
| picnic | 183.8 → 16.2 | 91.2% | 12.2 | 317 | 17.91 → 19.47 | 0.860 → 0.866 | 0.132 → 0.132 |
| pikachu | 11.5 → 2.8 | 75.8% | 1.5 | 58 | 20.92 → 21.36 | 0.917 → 0.918 | 0.140 → 0.135 |
| pipe | 75.2 → 15.1 | 79.9% | 14.1 | 178 | 15.57 → 18.55 | 0.743 → 0.783 | 0.314 → 0.300 |
| plant | 16.7 → 6.1 | 63.6% | 1.7 | 72 | 23.04 → 24.60 | 0.898 → 0.904 | 0.105 → 0.089 |
| roses | 27.3 → 7.1 | 73.8% | 2.2 | 89 | 15.76 → 20.23 | 0.815 → 0.849 | 0.164 → 0.112 |
| table | 42.8 → 2.5 | 94.1% | 3.0 | 112 | 22.27 → 27.21 | 0.959 → 0.984 | 0.089 → 0.035 |

Full numbers: `evaluation_report.md`, `.csv`, `.json`. These results were produced with a single γ = 0.1 for every scene (`02_evaluate_pipeline.py`); the per-scene values were added afterwards and have not been re-evaluated.

**How to read these numbers.**

- Image metrics are computed **inside the object mask** on **3 randomly sampled training frames** per scene, so they measure how much cleaner the object region becomes (mostly floaters in front of it removed). They are not held-out novel-view scores.
- The masks used for scoring come from the same SAM 2 pipeline that drives the vote.
- Renders are produced by OpenSplat's validation renderer after an 11-step resume, applied identically to the model before and after cleaning.
- There is no comparison against other distractor-removal methods yet.

## Usage

Requirements: Python 3.10+, PyTorch, `transformers`, SAM 2 (with the `sam2.1_hiera_large.pt` checkpoint), `plyfile`, `numpy`, `Pillow`, `psutil`, `torchmetrics`, `pyiqa`, and an OpenSplat binary.

```bash
# 1. train initial splats (OPENSPLAT_BIN can point to your binary)
./run_all.sh /path/to/nerfbusters

# 2. generate masks for every scene
./run_all_masks.sh /path/to/nerfbusters

# 3. extract one scene with a chosen gamma
python 01_extract_objects.py --scene_dir /path/to/nerfbusters/aloe \
  --ply_in /path/to/nerfbusters/opensplat_results/aloe_splat.ply \
  --ply_out /path/to/nerfbusters/extracted_results/aloe_extracted.ply --gamma 0.1

# 4. extract and evaluate all scenes (single gamma; use the _tuned script for per-scene gammas)
DATA_ROOT=/path/to/nerfbusters python 02_evaluate_pipeline.py
```

Each scene directory is expected to contain `images/` (or `images_4/`), `cameras.json`, `colmap/sparse`, and, after step 2, `masks/`.

## Limitations

- γ needs tuning per scene, and the tuned values have not yet been evaluated.
- The vote uses only each Gaussian's centre, ignoring its size and opacity, so large or semi-transparent Gaussians near the object boundary can be misclassified.
- The result depends on the quality of the text-prompted masks.
