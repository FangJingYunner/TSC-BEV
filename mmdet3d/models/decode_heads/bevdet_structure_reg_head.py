# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from mmcv.cnn import normal_init
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch import nn as nn
from mmdet.models import HEADS
# from ..builder import HEADS
from mmcv.cnn import ConvModule
import torch.nn as nn
import torch
from mmseg.models.builder import build_loss

@HEADS.register_module()
class BEVStructureRegHead(BaseModule):


    def __init__(self,
                 init_cfg=None,
                 bev_high_weight = 0.5,
                 in_channel =  128,
                 out_channel = 1,
                 use_mask = False):
        super(BEVStructureRegHead, self).__init__(init_cfg=init_cfg)

        self.out_channel = out_channel
        self.in_channel = in_channel
        self.bev_high_head = self.build_bev_high_head(self.out_channel)
        self.bev_high_weight = bev_high_weight
        self.use_mask = use_mask
    def build_bev_high_head(self, out_channel):


        layer = [

            ConvModule(self.in_channel, self.in_channel, kernel_size=3, padding=1, norm_cfg=dict(type='BN', requires_grad=True), stride=1),
            ConvModule(self.in_channel, out_channel, kernel_size=1, stride=1 , act_cfg=dict(type='Sigmoid'))
        ]

        return nn.Sequential(*layer)


    def bev_high_loss(self, perd_bev_high, gt_bev_high):
        gt_grid_mask, gt_grid_high = gt_bev_high

        batchsize, _, _, _ = perd_bev_high.shape
        if self.use_mask:
            loss = torch.sum(gt_grid_mask*(perd_bev_high-gt_grid_high)**2)/batchsize
        else:
            loss = torch.sum((perd_bev_high-gt_grid_high)**2)/batchsize
        return {'bev_high_loss': self.bev_high_weight * loss}

    def init_weights(self):
        """Initialize weights of classification layer."""
        super().init_weights()
        normal_init(self.bev_high_head, mean=0, std=0.01)

    @auto_fp16()
    def forward(self, *args,**kwargs):
        """Placeholder of forward function."""
        # if return_loss == true:
        return self.forward_train(*args,**kwargs)
        # else:
        #     return self.forward_test(**kwargs)


    def forward_train(self, bev_inputs, gt_bev_high):
        """Forward function for training.

        Args:
            inputs (list[torch.Tensor]): List of multi-level point features.
            img_metas (list[dict]): Meta information of each sample.
            pts_semantic_mask (torch.Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        bev_feat = self.bev_high_head(bev_inputs)
        losses = self.bev_high_loss(bev_feat, gt_bev_high)
        return bev_feat, losses

    def forward_test(self, inputs):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level point features.
            img_metas (list[dict]): Meta information of each sample.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.bev_high_head(inputs)

