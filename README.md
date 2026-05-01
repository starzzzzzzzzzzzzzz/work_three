# Patch-Score Artifact Probe

Exploratory scripts for testing whether LLaVA's CLIP/ViT vision tower shows
background-dominated patch-score artifacts on a small COCO subset.

## Run

```bash
chmod +x scripts/run_patch_artifact_probe_50.sh
./scripts/run_patch_artifact_probe_50.sh
```

For a larger exploratory scan:

```bash
chmod +x scripts/run_patch_artifact_probe_500.sh
./scripts/run_patch_artifact_probe_500.sh
```

The script only loads the LLaVA vision tower weights from the local HF
safetensors shards. It does not load the 7B language model and does not generate
captions.

The default sampler keeps images with one or two non-crowd COCO boxes whose
dominant box covers 5%-65% of the image, then samples clear foreground cases.
The 50-image wrapper saves overlays for quick manual inspection. The 500-image
wrapper skips overlays by default and writes only CSV/JSON/NPZ artifacts.

## Outputs

The default output directory is `runs/patch_artifact_probe_50/` or
`runs/patch_artifact_probe_500/`, depending on the wrapper.

- `patch_artifact_metrics.csv`: one row per sampled image, sorted by artifact strength.
- `summary.json`: aggregate top-1 background rate and top-k background ratios.
- `arrays/*.npz`: patch-score, low-pass stability-score, and foreground mask arrays.
- `overlays/*.png`: image / PatchScore heatmap / COCO foreground mask for manual inspection.

## Main Signals

- `patch_point_in_box`: whether the highest PatchScore patch lies inside a COCO foreground box.
- `patch_top1_is_background`: whether the highest PatchScore patch is background.
- `patch_topk_background_ratio`: background fraction among the top-k PatchScore patches.
- `patch_fg_bg_gap`: foreground mean PatchScore minus background mean PatchScore.
- `stability_topk_background_ratio`: background fraction among low-pass-stability top-k patches.
