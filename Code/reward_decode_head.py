import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

import numpy as np

from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
from ..builder import build_loss
from ..losses import accuracy
from ..utils import resize
from mmseg.registry import MODELS


class RewardDecodeHead(BaseDecodeHead):
    """
        Extends the Base Decode Head to implement the logif for the reward-based loss computation
    """
    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 out_channels=None,
                 threshold=None,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=dict(type='Normal', std=0.01, override=dict(name='conv_seg'))):
        # Initialize the base class
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            out_channels=out_channels,
            threshold=threshold,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            in_index=in_index,
            input_transform=input_transform,
            loss_decode=loss_decode,
            ignore_index=ignore_index,
            sampler=sampler,
            align_corners=align_corners,
            init_cfg=init_cfg
        )
        
        self.mIoU_metric_loss = build_loss(dict(
            type='IoUMetricLoss', num_classes=self.num_classes, ignore_index=self.ignore_index))
    
        self.reinforce_loss = build_loss(dict(
            type='CrossEntropyRewardLoss'))
        
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList, train_cfg: ConfigType) -> dict:
        """
        Overriding the loss method to incorporate the reward-based loss computation.
        """
        seg_logits = self.forward(inputs)
        
        mIoU_each_photo = self.reward(seg_logits, batch_data_samples)
        mean_mIoU_other_photos = (np.sum(mIoU_each_photo) - mIoU_each_photo) / (len(mIoU_each_photo) - 1)
        reward_vector_self_baseline = mIoU_each_photo - 0.1 * mean_mIoU_other_photos
        
        losses = self.loss_by_feat_reinforce(seg_logits, batch_data_samples, reward_vector_self_baseline)
        
        return losses
    
    def reward(self, seg_logits: Tensor, batch_data_samples: SampleList) -> np.ndarray:
        """
        Compute the IoU metric for each sample in the batch and return as a reward signal.
        """
        seg_label = self._stack_batch_gt(batch_data_samples)
        seg_logits = resize(input=seg_logits, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1)
        reward_values = np.zeros(seg_logits.shape[0])
        
        for i in range(seg_logits.shape[0]):
            reward_values[i] = self.mIoU_metric_loss(seg_logits[i:i+1], seg_label[i:i+1])
        
        return reward_values
    
    def loss_by_feat_reinforce(self, seg_logits: Tensor, batch_data_samples: SampleList, reward: np.ndarray) -> dict:
        """
        Extend the base class loss_by_feat to use reinforcement learning-specific loss functions.
        """
        seg_label = self._stack_batch_gt(batch_data_samples)
        seg_logits = resize(input=seg_logits, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        loss = dict()
        loss[self.reinforce_loss.loss_name] = self.reinforce_loss(seg_logits, seg_label, weight=seg_weight, ignore_index=self.ignore_index, reward=reward)
        loss['acc_seg'] = accuracy(seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss