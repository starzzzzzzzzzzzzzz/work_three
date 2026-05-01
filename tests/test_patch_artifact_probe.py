import numpy as np

from scripts.patch_artifact_probe import (
    bbox_to_patch_mask,
    compute_patch_artifact_metrics,
    normalize_bbox,
)


def test_normalize_bbox_clips_xywh_to_image_bounds():
    assert normalize_bbox([-5, 10, 30, 100], width=100, height=80) == (0.0, 10.0, 25.0, 80.0)
    assert normalize_bbox([90, 70, 50, 50], width=100, height=80) == (90.0, 70.0, 100.0, 80.0)


def test_bbox_to_patch_mask_marks_overlapping_patch_centers():
    mask = bbox_to_patch_mask([(25, 25, 75, 75)], width=100, height=100, grid_size=4)

    expected = np.array(
        [
            [False, False, False, False],
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, False],
        ]
    )
    np.testing.assert_array_equal(mask, expected)


def test_compute_patch_artifact_metrics_summarizes_background_dominance():
    scores = np.array(
        [
            [0.10, 0.20, 0.95],
            [0.15, 0.70, 0.80],
            [0.05, 0.30, 0.40],
        ],
        dtype=np.float32,
    )
    fg_mask = np.array(
        [
            [False, False, False],
            [False, True, True],
            [False, False, False],
        ]
    )

    metrics = compute_patch_artifact_metrics(scores, fg_mask, topk=3)

    assert metrics["top1_is_background"] is True
    assert metrics["top1_patch_index"] == 2
    assert metrics["topk_background_ratio"] == 1 / 3
    assert metrics["point_in_box"] == 0.0
    assert metrics["fg_bg_gap"] > 0
