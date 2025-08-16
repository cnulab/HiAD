"""Anomaly metrics."""
import numpy as np
import warnings
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAveragePrecision
from torchmetrics.utilities.compute import auc
from torchmetrics.classification.roc import BinaryROC
from torchmetrics.classification import BinaryPrecisionRecallCurve as _BinaryPrecisionRecallCurve
from torch import Tensor
from torchmetrics.functional.classification.precision_recall_curve import (
    _adjust_threshold_arg,
    _binary_precision_recall_curve_update,
)
from .pro import AUPRO
warnings.filterwarnings('ignore')

class BinaryPrecisionRecallCurve(_BinaryPrecisionRecallCurve):
    """Binary precision-recall curve with without threshold prediction normalization."""

    @staticmethod
    def _binary_precision_recall_curve_format(
        preds,
        target,
        thresholds = None,
        ignore_index = None,
    ) :
        """Similar to torchmetrics' ``_binary_precision_recall_curve_format`` except it does not apply sigmoid."""
        preds = preds.flatten()
        target = target.flatten()
        if ignore_index is not None:
            idx = target != ignore_index
            preds = preds[idx]
            target = target[idx]

        thresholds = _adjust_threshold_arg(thresholds, preds.device)
        return preds, target, thresholds

    def update(self, preds, target) -> None:
        """Update metric state with new predictions and targets.

        Unlike the base class, this accepts raw predictions and targets.

        Args:
            preds (Tensor): Predicted probabilities
            target (Tensor): Ground truth labels
        """
        preds, target, _ = BinaryPrecisionRecallCurve._binary_precision_recall_curve_format(
            preds,
            target,
            self.thresholds,
            self.ignore_index,
        )
        state = _binary_precision_recall_curve_update(preds, target, self.thresholds)
        if isinstance(state, Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])


def compute_imagewise_metrics_gpu(
    prediction_scores, gt_labels, device = None, **kwargs
):
    """
       Computes image-wise metrics (Image Auroc) on GPU.

       Args:
           prediction_scores: [np.array or list] [N] Assignment weights
                                       per image. Higher indicates higher
                                       probability of being an anomaly.
           gt_labels: [np.array or list] [N] Binary labels - 1
                                       if image is an anomaly, 0 if not.
           device: torch.device
       """
    if device is None:
        device = torch.device("cuda")
    prediction_scores = torch.tensor(prediction_scores).to(device)
    gt_labels = torch.tensor(gt_labels).to(device).long()

    fpr, tpr, thresholds = BinaryROC(thresholds=None)(
         prediction_scores, gt_labels
    )
    auroc = auc(fpr, tpr, reorder=True)
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = thresholds[maxindex]

    return {"image_auroc": auroc.item(), "image_threshold": threshold.item()}


def compute_pixelwise_metrics_gpu(prediction_masks, gt_masks, device =None, **kwargs):
    """
     Computes pixel-wise metrics (Pixel-AUROC, AP, F1) on GPU

     Args:
         prediction_masks: [list of np.arrays or np.array] [NxHxW] Contains
                                 generated segmentation masks.
         gt_masks: [list of np.arrays or np.array] [NxHxW] Contains
                             predefined ground truth segmentation masks
         device: torch.device
    """

    if device is None:
        device = torch.device("cuda")

    if isinstance(prediction_masks, list):
        prediction_masks = np.stack(prediction_masks)

    if isinstance(gt_masks, list):
        gt_masks = np.stack(gt_masks)

    prediction_masks = torch.tensor(prediction_masks).to(device)
    gt_masks = torch.tensor(gt_masks).to(device).long()

    flat_anomaly_segmentations = prediction_masks.flatten()
    flat_ground_truth_masks = gt_masks.flatten()

    fpr, tpr, _ = BinaryROC(thresholds=None)(
        flat_anomaly_segmentations, flat_ground_truth_masks
    )

    auroc = auc(fpr, tpr, reorder=True)

    precision, recall, thresholds = BinaryPrecisionRecallCurve()(flat_anomaly_segmentations,flat_ground_truth_masks)
    f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
    seg_threshold = thresholds[torch.argmax(f1_score)]
    f1 = torch.max(f1_score)

    ap = BinaryAveragePrecision(thresholds=None)(flat_anomaly_segmentations, flat_ground_truth_masks)

    return {
        "pixel_auroc": auroc.item(),
        "pixel_ap": ap.item(),
        "pixel_f1": f1.item(),
        "seg_threshold": seg_threshold.item()
    }


def compute_pro_gpu(prediction_masks, gt_masks, device = None, **kwargs):
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR on GPU
       Args:
           prediction_masks (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
           gt_masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
           device: torch.device
       """
    if device is None:
        device = torch.device("cuda")
    prediction_masks = torch.tensor(prediction_masks).to(device)
    gt_masks = torch.tensor(gt_masks).to(device).long()
    aupro = AUPRO(fpr_limit=0.3, num_thresholds=200)(prediction_masks, gt_masks)
    return {"pixel_pro": aupro.item()}



def compute_pixelwise_metrics_gpu_resize(prediction_masks, gt_masks, resize, device = None, **kwargs):
    if device is None:
        device = torch.device("cuda")
    prediction_masks_resize = F.interpolate(torch.tensor(prediction_masks[:,None,:,:]),
                                         (resize, resize), mode = 'bilinear').to(device).squeeze(1)
    gt_masks_resize = F.interpolate(torch.tensor(gt_masks[:, None, :, :]),
                                         (resize, resize), mode='nearest').to(device).squeeze(1)
    return compute_pixelwise_metrics_gpu(prediction_masks_resize, gt_masks_resize, device)


def compute_pro_gpu_resize(prediction_masks, gt_masks, resize,device = None, **kwargs):
    if device is None:
        device = torch.device("cuda")
    prediction_masks_resize = F.interpolate(torch.tensor(prediction_masks[:, None, :, :]),
                                            (resize, resize), mode='bilinear').to(device).squeeze(1)
    gt_masks_resize = F.interpolate(torch.tensor(gt_masks[:, None, :, :]),
                                    (resize, resize), mode='nearest').to(device).squeeze(1)
    return compute_pro_gpu(prediction_masks_resize, gt_masks_resize, device)
