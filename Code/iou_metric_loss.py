import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmseg.registry import MODELS
#from .utils import weight_loss


@MODELS.register_module()
class IoUMetricLoss(nn.Module):
    def __init__(self, num_classes, ignore_index = -1, loss_name= 'loss_iou_metric'):
        super(IoUMetricLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def batch_intersection_union(self, pred, target):
        results = []
        for i in range(pred.shape[0]):
            results.append(self.intersect_and_union(pred[i], target[i]))
        return results
    
    def intersect_and_union(self, pred_label: torch.tensor, label: torch.tensor):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != self.ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(self.num_classes), min=0,
            max=self.num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(self.num_classes), min=0,
            max=self.num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(self.num_classes), min=0,
            max=self.num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label
    
    def compute_total(self, results):
        results = tuple(zip(*results))
        #print(len(results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        
        return total_area_intersect, total_area_union, total_area_pred_label, total_area_label
    
    def iou(self, total_area_intersect, total_area_union, total_area_pred_label, total_area_label):
        """Calculate Intersection over Union.

        Args:
            total_area_intersect (torch.Tensor): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (torch.Tensor): The union of prediction and ground
                truth histogram on all classes.

        Returns:
            torch.Tensor: Intersection over Union.
        """
        iou = total_area_intersect / total_area_union#(total_area_union + 1e-10)
        result = np.nanmean(iou)
        
        if np.isnan(result):
            return 0.5
        else:
            return result
    
    def forward(self, pred_label, label):
        pred = torch.argmax(pred_label, dim=1)
        results = self.batch_intersection_union(pred, label)

        a, b, c, d = self.compute_total(results)

        corrected_loss = 1 - self.iou(a, b, c, d)
        
        return corrected_loss
         
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name