"""Anomaly metrics."""
import numpy as np
from sklearn import metrics
from numpy import ndarray
import pandas as pd
from skimage import measure
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
import torch.nn.functional as F



def compute_imagewise_metrics(
    prediction_scores, gt_labels, **kwargs
):
    """
    Computes image-wise metrics (Image Auroc).

    Args:
        prediction_scores: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        gt_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        gt_labels, prediction_scores
    )

    auroc = metrics.roc_auc_score(
        gt_labels, prediction_scores
    )

    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = thresholds[maxindex]

    return {"image_auroc": auroc, "image_threshold": threshold}



def compute_pixelwise_metrics(prediction_masks, gt_masks, **kwargs):
    """
    Computes pixel-wise metrics (Pixel-AUROC, AP, F1) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        prediction_masks: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        gt_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(prediction_masks, list):
        prediction_masks = np.stack(prediction_masks)
    if isinstance(gt_masks, list):
        gt_masks = np.stack(gt_masks)

    flat_anomaly_segmentations = prediction_masks.ravel()
    flat_ground_truth_masks = gt_masks.ravel()

    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = precision_recall_curve(flat_ground_truth_masks, flat_anomaly_segmentations)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    seg_threshold = thresholds[np.argmax(f1)]

    f1 = f1[np.argmax(f1)]
    ap = average_precision_score(flat_ground_truth_masks, flat_anomaly_segmentations)

    return {
        "pixel_auroc": auroc,
        "pixel_ap": ap,
        "pixel_f1": f1,
        "seg_threshold": seg_threshold
    }


def compute_pro(prediction_masks, gt_masks, num_th: int = 200, **kwargs):

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        prediction_masks (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        gt_masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """
    assert isinstance(prediction_masks, ndarray), "type(amaps) must be ndarray"
    assert isinstance(gt_masks, ndarray), "type(masks) must be ndarray"
    assert prediction_masks.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert gt_masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert prediction_masks.shape == gt_masks.shape, "amaps.shape and masks.shape must be same"
    assert set(gt_masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(prediction_masks, dtype=bool)
    min_th = prediction_masks.min()
    max_th = prediction_masks.max()
    delta = (max_th - min_th) / num_th
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[prediction_masks <= th] = 0
        binary_amaps[prediction_masks > th] = 1
        pros = []
        for binary_amap, mask in zip(binary_amaps, gt_masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)
        inverse_masks = 1 - gt_masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        df_new = pd.DataFrame([{"pro": np.mean(pros), "fpr": fpr, "threshold": th}])
        df = pd.concat([df, df_new], ignore_index=True)
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()
    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return {"pixel_pro":pro_auc}


def compute_pixelwise_metrics_resize(prediction_masks, gt_masks, resize, **kwargs):
    prediction_masks_resize = F.interpolate(torch.tensor(prediction_masks[:,None,:,:]),
                                         (resize, resize), mode = 'bilinear').numpy().squeeze(1)
    gt_masks_resize = F.interpolate(torch.tensor(gt_masks[:, None, :, :]),
                                         (resize, resize), mode='nearest').numpy().squeeze(1)
    return compute_pixelwise_metrics(prediction_masks_resize, gt_masks_resize)


def compute_pro_resize(prediction_masks, gt_masks, resize, **kwargs):
    prediction_masks_resize = F.interpolate(torch.tensor(prediction_masks[:,None,:,:]),
                                         (resize, resize), mode = 'bilinear').numpy().squeeze(1)
    gt_masks_resize = F.interpolate(torch.tensor(gt_masks[:, None, :, :]),
                                         (resize, resize), mode='nearest').numpy().squeeze(1)
    return compute_pro(prediction_masks_resize, gt_masks_resize)
