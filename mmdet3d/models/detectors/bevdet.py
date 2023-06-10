# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from ..decode_heads.decode_head import Base3DDecodeHead
import math

@DETECTORS.register_module()
class BEVDet(CenterPoint):
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self, img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, **kwargs):
        super(BEVDet, self).__init__(**kwargs)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = \
            builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        x = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:7])
        x = self.bev_encoder(x)
        return [x], depth

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth, bev_feat_list = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth, bev_feat_list)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs


@DETECTORS.register_module()
class BEVDetTRT(BEVDet):

    def result_serialize(self, outs):
        outs_ = []
        for out in outs:
            for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                outs_.append(out[0][key])
        return outs_

    def result_deserialize(self, outs):
        outs_ = []
        keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for head_id in range(len(outs) // 6):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 6 + kid]
            outs_.append(outs_head)
        return outs_

    def forward(
        self,
        img,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
    ):
        x = self.img_backbone(img)
        x = self.img_neck(x)
        x = self.img_view_transformer.depth_net(x)
        depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
        tran_feat = x[:, self.img_view_transformer.D:(
            self.img_view_transformer.D +
            self.img_view_transformer.out_channels)]
        tran_feat = tran_feat.permute(0, 2, 3, 1)
        x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
                               ranks_depth, ranks_feat, ranks_bev,
                               interval_starts, interval_lengths)
        x = x.permute(0, 3, 1, 2).contiguous()
        bev_feat = self.bev_encoder(x)
        outs = self.pts_bbox_head([bev_feat])
        outs = self.result_serialize(outs)
        return outs

    def get_bev_pool_input(self, input):
        coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
        return self.img_view_transformer.voxel_pooling_prepare_v2(coor)


@DETECTORS.register_module()
class BEVDet4D(BEVDet):
    r"""BEVDet4D paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2203.17054>`_

    Args:
        pre_process (dict | None): Configuration dict of BEV pre-process net.
        align_after_view_transfromation (bool): Whether to align the BEV
            Feature after view transformation. By default, the BEV feature of
            the previous frame is aligned during the view transformation.
        num_adj (int): Number of adjacent frames.
        with_prev (bool): Whether to set the BEV feature of previous frame as
            all zero. By default, False.
    """
    def __init__(self,
                 pre_process=None,
                 align_after_view_transfromation=False,
                 num_adj=1,
                 with_prev=True,
                 **kwargs):
        super(BEVDet4D, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1

        self.with_prev = with_prev

    @force_fp32()
    def shift_feature(self, input, trans, rots, bda, bda_adj=None):
        n, c, h, w = input.shape
        _, v, _ = trans[0].shape

        # generate grid
        xs = torch.linspace(
            0, w - 1, w, dtype=input.dtype,
            device=input.device).view(1, w).expand(h, w)
        ys = torch.linspace(
            0, h - 1, h, dtype=input.dtype,
            device=input.device).view(h, 1).expand(h, w)
        grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
        grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)

        # get transformation from current ego frame to adjacent ego frame
        # transformation from current camera frame to current ego frame
        c02l0 = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        c02l0[:, :, :3, :3] = rots[0][:, 0:1, :, :]
        c02l0[:, :, :3, 3] = trans[0][:, 0:1, :]
        c02l0[:, :, 3, 3] = 1

        # transformation from adjacent camera frame to current ego frame
        c12l0 = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        c12l0[:, :, :3, :3] = rots[1][:, 0:1, :, :]
        c12l0[:, :, :3, 3] = trans[1][:, 0:1, :]
        c12l0[:, :, 3, 3] = 1

        # add bev data augmentation
        bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        bda_[:, :, :3, :3] = bda.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        c02l0 = bda_.matmul(c02l0)
        if bda_adj is not None:
            bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
            bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        c12l0 = bda_.matmul(c12l0)

        # transformation from current ego frame to adjacent ego frame
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(
            n, 1, 1, 4, 4)
        '''
          c02l0 * inv(c12l0)
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        l02l1 = l02l1[:, :, :,
                      [True, True, False, True], :][:, :, :, :,
                                                    [True, True, False, True]]

        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1,
                                                            2) * 2.0 - 1.0
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)
        return output

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input, gt_depth):
        x = self.image_encoder(img)
        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input, gt_depth])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth

    def extract_img_feat_sequential(self, inputs, feat_prev):
        imgs, rots_curr, trans_curr, intrins = inputs[:4]
        rots_prev, trans_prev, post_rots, post_trans, bda = inputs[4:]
        bev_feat_list = []
        mlp_input = self.img_view_transformer.get_mlp_input(
            rots_curr[0:1, ...], trans_curr[0:1, ...], intrins, post_rots,
            post_trans, bda[0:1, ...])
        inputs_curr = (imgs, rots_curr[0:1, ...], trans_curr[0:1, ...],
                       intrins, post_rots, post_trans, bda[0:1,
                                                           ...], mlp_input)
        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
        bev_feat_list.append(bev_feat)

        # align the feat_prev
        _, C, H, W = feat_prev.shape
        feat_prev = \
            self.shift_feature(feat_prev,
                               [trans_curr, trans_prev],
                               [rots_curr, rots_prev],
                               bda)
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        B, N, _, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
        extra = [
            rots.view(B, self.num_frame, N, 3, 3),
            trans.view(B, self.num_frame, N, 3),
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        return imgs, rots, trans, intrins, post_rots, post_trans, bda

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'])
        imgs, rots, trans, intrins, post_rots, post_trans, bda = \
            self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only
        for img, rot, tran, intrin, post_rot, post_tran in zip(
                imgs, rots, trans, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    rot, tran = rots[0], trans[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    rots[0], trans[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, rot, tran, intrin, post_rot,
                               post_tran, bda, mlp_input, kwargs['gt_depth'])#rot,tran是从sweepsensor2keyego
                if key_frame:
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False
        if pred_prev:
            assert self.align_after_view_transfromation
            assert rots[0].shape[0] == 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            trans_curr = trans[0].repeat(self.num_frame - 1, 1, 1)
            rots_curr = rots[0].repeat(self.num_frame - 1, 1, 1, 1)
            trans_prev = torch.cat(trans[1:], dim=0)
            rots_prev = torch.cat(rots[1:], dim=0)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
            return feat_prev, [
                imgs[0], rots_curr, trans_curr, intrins[0], rots_prev,
                trans_prev, post_rots[0], post_trans[0], bda_curr
            ]
        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [trans[0], trans[adj_id]],
                                       [rots[0], rots[adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        x = self.bev_encoder(bev_feat)
        return [x], depth_list[0], bev_feat_list


@DETECTORS.register_module()
class BEVDepth4D(BEVDet4D):

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth, bev_feats_list = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses




@DETECTORS.register_module()
class BEVDepth4DHigh(BEVDet4D):

    def __init__(self, bev_structure_reg_head=None, depth_loss_type="v1", **kwargs):
        super(BEVDepth4DHigh, self).__init__(**kwargs)

        self.depth_loss_type = depth_loss_type
        # if bev_structure_reg_head is not None:
        #     self.bev_structure_reg_head = builder.build_head(bev_structure_reg_head)
        if self.train_cfg is not None and self.train_cfg['object_temporal_consistance_loss'] is not None:
            self.temporal_consistance_weight = self.train_cfg['object_temporal_consistance_loss']['weight']
            self.before_fusion = self.train_cfg['object_temporal_consistance_loss']['before_fusion']
            self.lidar_guided = self.train_cfg['object_temporal_consistance_loss']['lidar_guided']
            self.use_half_feature = self.train_cfg['object_temporal_consistance_loss']['use_half_feature']
            self.bilinear_interpolate = self.train_cfg['object_temporal_consistance_loss']['bilinear_interpolate']

        self.tcl_start_flag = False

        # if auxiliary_pts_bbox_head:
        #     pts_train_cfg = kwargs["train_cfg"].pts if kwargs["train_cfg"] else None
        #     auxiliary_pts_bbox_head.update(train_cfg=pts_train_cfg)
        #     pts_test_cfg = kwargs["test_cfg"].pts if kwargs["test_cfg"] else None
        #     auxiliary_pts_bbox_head.update(test_cfg=pts_test_cfg)
        #     self.auxiliary_pts_bbox_head = builder.build_head(auxiliary_pts_bbox_head)


    def temporal_consistance_loss(self, img_feats, bev_feats_list, corner_relation_list, corner_mask_list):
    # def temporal_consistance_loss(self, img_feats, bev_feats_list, corner_relation_list):

        bev_feats_list = list(map(list, zip(*bev_feats_list)))
        # total_cos_loss = torch.zeros()
        total_cos_loss = img_feats.new_zeros(9, 9)
        batchsize = len(bev_feats_list)
        # if self.before_fusion:
        #     for batch in range(len(corner_relation_list)):
        #         corner_relation_list[batch] = corner_relation_list[batch][:,:,2:]

        # corner_relation = torch.split(corner_relation[0], 2, dim=2)
        for bev_feat, prev_bev_feat_list,corner_relation, corner_mask in zip(img_feats, bev_feats_list, corner_relation_list, corner_mask_list):
        # for bev_feat, prev_bev_feat_list,corner_relation in zip(img_feats, bev_feats_list, corner_relation_list):


            # 判断历史帧是否用到
            prev_bev_feat_use = torch.zeros(len(prev_bev_feat_list))
            for prev_frame_id, prev_bev_feat in enumerate(prev_bev_feat_list):
                if prev_bev_feat.max() == 0:
                    prev_bev_feat_use[prev_frame_id] = 0
                else:
                    prev_bev_feat_use[prev_frame_id] = 1

            if self.bilinear_interpolate:
                bev_feat = bev_feat.unsqueeze(0)
                _, C, H, W = bev_feat.shape

            for bbox_corner_relation, bbox_corner_mask in zip(corner_relation, corner_mask):
            # for bbox_corner_relation in corner_relation:

                curr_corner_relation = bbox_corner_relation[:, :2]
                if curr_corner_relation.min() <= 0:
                    continue
                if self.bilinear_interpolate:
                    curr_corner_relation = curr_corner_relation.unsqueeze(0).unsqueeze(0)
                    curr_corner_relation_norm = (2 * curr_corner_relation / W) - 1
                    bilinear_current_feat = F.grid_sample(bev_feat,curr_corner_relation_norm,mode='bilinear')
                    bilinear_current_feat = bilinear_current_feat.squeeze(2).squeeze(0)
                    if self.before_fusion == False and self.use_half_feature == True:
                        bilinear_current_feat = bilinear_current_feat[:int(C/2),:]

                    curr_norm_tensor = torch.norm(bilinear_current_feat, dim=0)
                    curr_box_cos_sim = torch.mm(bilinear_current_feat.t(), bilinear_current_feat) / torch.mm(
                        curr_norm_tensor.unsqueeze(1), curr_norm_tensor.unsqueeze(0))
                else:
                    if self.before_fusion ==False and self.use_half_feature == True:
                        C,_,_ = bev_feat.shape
                        curr_feat_tensor = bev_feat[:int(C/2), curr_corner_relation[:, 0].long(), curr_corner_relation[:, 1].long()]
                        curr_norm_tensor = torch.norm(curr_feat_tensor, dim=0)
                    else:
                        curr_feat_tensor = bev_feat[:, curr_corner_relation[:, 0].long(), curr_corner_relation[:, 1].long()]
                        curr_norm_tensor = torch.norm(curr_feat_tensor, dim=0)

                    curr_box_cos_sim = torch.mm(curr_feat_tensor.t(), curr_feat_tensor) / torch.mm(
                        curr_norm_tensor.unsqueeze(1), curr_norm_tensor.unsqueeze(0))



                box_total_cos_loss_sum = img_feats[0].new_zeros(9, 9)
                if self.lidar_guided:
                    box_frame_count = img_feats.new_zeros(9, 9)
                else:
                    box_frame_count = 0

                prev_bbox_corners_list = torch.split(bbox_corner_relation, 2, dim=1)
                prev_bbox_corner_mask_list = torch.split(bbox_corner_mask, 1, dim=0)

                for prev_frame_id, (prev_bbox_corner, prev_bbox_corner_mask) in enumerate(zip(prev_bbox_corners_list, prev_bbox_corner_mask_list)):
                # for prev_frame_id, prev_bbox_corner in enumerate(prev_bbox_corners_list):

                    # 历史帧中没有该障碍物 or 障碍物出界了
                    if prev_bbox_corner.max() == 0 or prev_bbox_corner.min() == -1:
                        continue

                    # 当前历史帧没用到，当前帧之前的历史帧也没用到
                    if prev_bev_feat_use[prev_frame_id] == 0:
                        break

                    if self.bilinear_interpolate:
                        prev_bbox_corner = prev_bbox_corner.unsqueeze(0).unsqueeze(0)
                        prev_bev_feat = prev_bev_feat_list[prev_frame_id].unsqueeze(0)
                        _,C,H,W = prev_bev_feat.shape
                        prev_bbox_corner_norm = (2 * prev_bbox_corner / W) - 1
                        bilinear_prev_feat = F.grid_sample(bev_feat,prev_bbox_corner_norm,mode='bilinear')
                        bilinear_prev_feat = bilinear_prev_feat.squeeze(2).squeeze(0)
                        if self.before_fusion == False and self.use_half_feature == True:
                            bilinear_prev_feat = bilinear_prev_feat[:int(C/2),:]

                        prev_norm_tensor = torch.norm(bilinear_prev_feat, dim=0)
                        prev_box_cos_sim = torch.mm(bilinear_prev_feat.t(), bilinear_prev_feat) / torch.mm(
                            prev_norm_tensor.unsqueeze(1), prev_norm_tensor.unsqueeze(0))
                    else:

                        prev_feat_tensor = prev_bev_feat_list[prev_frame_id][:, prev_bbox_corner[:, 0].long(), prev_bbox_corner[:, 1].long()]

                        prev_norm_tensor = torch.norm(prev_feat_tensor, dim=0)

                        prev_box_cos_sim = torch.mm(prev_feat_tensor.t(), prev_feat_tensor) / torch.mm(
                            prev_norm_tensor.unsqueeze(1), prev_norm_tensor.unsqueeze(0))

                    if self.lidar_guided:
                        box_total_cos_loss_sum += prev_bbox_corner_mask.squeeze(0) * abs(curr_box_cos_sim-prev_box_cos_sim)
                        box_frame_count += prev_bbox_corner_mask.squeeze(0)
                    else:
                        box_total_cos_loss_sum += abs(curr_box_cos_sim-prev_box_cos_sim)
                        box_frame_count += 1

                if self.lidar_guided:
                    if box_frame_count.max() > 0:
                            total_cos_loss += box_total_cos_loss_sum/(box_frame_count+1e-6)
                else:
                    if box_frame_count > 0:
                        total_cos_loss += box_total_cos_loss_sum / box_frame_count

        total_cos_loss_sum = self.temporal_consistance_weight*(torch.sum(total_cos_loss)/81/batchsize)

        return {'box_cos_sim_loss': total_cos_loss_sum}
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      gt_bev_high=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        img_feats, pts_feats, depth, bev_feats_list = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)

        if self.depth_loss_type == "v1":
            gt_depth = kwargs['gt_depth']
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
            losses = dict(loss_depth=loss_depth)
        elif self.depth_loss_type == "v2":
            gt_depth = kwargs['gt_depth']
            loss_depth = self.img_view_transformer.get_depth_loss_v2(gt_depth, depth)
            losses = dict(loss_depth=loss_depth)
        elif self.depth_loss_type == "none":
            losses = dict()

        # losses = dict()

        # if hasattr(self, 'bev_structure_reg_head'):
        #     bev_high_feature, bev_high_loss = self.bev_structure_reg_head(bev_feats_list, gt_bev_high)
        #     losses.update(bev_high_loss)
        #     img_feats[0] = torch.cat([img_feats[0], bev_high_feature], dim=1)
        #
        if self.tcl_start_flag:
            corner_relation = kwargs['corner_relation']
            corner_mask = kwargs['corner_mask']

            if self.before_fusion:
                temporal_consistance_loss = self.temporal_consistance_loss(bev_feats_list[0], bev_feats_list, corner_relation,corner_mask)
                # temporal_consistance_loss = self.temporal_consistance_loss(bev_feats_list[0], bev_feats_list, corner_relation)

            else:
                temporal_consistance_loss = self.temporal_consistance_loss(img_feats[0], bev_feats_list, corner_relation,corner_mask)
                # temporal_consistance_loss = self.temporal_consistance_loss(img_feats[0], bev_feats_list, corner_relation)

            losses.update(temporal_consistance_loss)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

@DETECTORS.register_module()
class BEVDepthTCAuxiliary(BEVDet4D):

    def __init__(self, bev_structure_reg_head=None, auxiliary_pts_bbox_head=None, depth_loss_type="v1", **kwargs):
        super(BEVDepthTCAuxiliary, self).__init__(**kwargs)

        self.depth_loss_type = depth_loss_type
        # if bev_structure_reg_head is not None:
        #     self.bev_structure_reg_head = builder.build_head(bev_structure_reg_head)
        if self.train_cfg is not None and self.train_cfg['object_temporal_consistance_loss'] is not None:
            self.temporal_consistance_weight = self.train_cfg['object_temporal_consistance_loss']['weight']
            self.before_fusion = self.train_cfg['object_temporal_consistance_loss']['before_fusion']
            self.lidar_guided = self.train_cfg['object_temporal_consistance_loss']['lidar_guided']
            self.use_half_feature = self.train_cfg['object_temporal_consistance_loss']['use_half_feature']
            self.bilinear_interpolate = self.train_cfg['object_temporal_consistance_loss']['bilinear_interpolate']

        self.tcl_start_flag = False

        if auxiliary_pts_bbox_head:
            pts_train_cfg = kwargs["train_cfg"].pts if kwargs["train_cfg"] else None
            auxiliary_pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = kwargs["test_cfg"].pts if kwargs["test_cfg"] else None
            auxiliary_pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.auxiliary_pts_bbox_head = builder.build_head(auxiliary_pts_bbox_head)
            # self.auxiliary_weight = kwargs["train_cfg"].auxiliary_weight if kwargs["train_cfg"].auxiliary_weight else 0.125

    def sweeps_forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.auxiliary_pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.auxiliary_pts_bbox_head.loss(*loss_inputs)
        return losses


    def temporal_consistance_loss(self, img_feats, bev_feats_list, corner_relation_list, corner_mask_list):
    # def temporal_consistance_loss(self, img_feats, bev_feats_list, corner_relation_list):

        bev_feats_list = list(map(list, zip(*bev_feats_list)))
        # total_cos_loss = torch.zeros()
        total_cos_loss = img_feats.new_zeros(9, 9)
        batchsize = len(bev_feats_list)
        # if self.before_fusion:
        #     for batch in range(len(corner_relation_list)):
        #         corner_relation_list[batch] = corner_relation_list[batch][:,:,2:]

        # corner_relation = torch.split(corner_relation[0], 2, dim=2)
        for bev_feat, prev_bev_feat_list,corner_relation, corner_mask in zip(img_feats, bev_feats_list, corner_relation_list, corner_mask_list):
        # for bev_feat, prev_bev_feat_list,corner_relation in zip(img_feats, bev_feats_list, corner_relation_list):


            # 判断历史帧是否用到
            prev_bev_feat_use = torch.zeros(len(prev_bev_feat_list))
            for prev_frame_id, prev_bev_feat in enumerate(prev_bev_feat_list):
                if prev_bev_feat.max() == 0:
                    prev_bev_feat_use[prev_frame_id] = 0
                else:
                    prev_bev_feat_use[prev_frame_id] = 1

            if self.bilinear_interpolate:
                bev_feat = bev_feat.unsqueeze(0)
                _, C, H, W = bev_feat.shape

            for bbox_corner_relation, bbox_corner_mask in zip(corner_relation, corner_mask):
            # for bbox_corner_relation in corner_relation:

                curr_corner_relation = bbox_corner_relation[:, :2]
                if curr_corner_relation.min() <= 0:
                    continue
                if self.bilinear_interpolate:
                    curr_corner_relation = curr_corner_relation.unsqueeze(0).unsqueeze(0)
                    curr_corner_relation_norm = (2 * curr_corner_relation / W) - 1
                    bilinear_current_feat = F.grid_sample(bev_feat,curr_corner_relation_norm,mode='bilinear')
                    bilinear_current_feat = bilinear_current_feat.squeeze(2).squeeze(0)
                    if self.before_fusion == False and self.use_half_feature == True:
                        bilinear_current_feat = bilinear_current_feat[:int(C/2),:]

                    curr_norm_tensor = torch.norm(bilinear_current_feat, dim=0)
                    curr_box_cos_sim = torch.mm(bilinear_current_feat.t(), bilinear_current_feat) / torch.mm(
                        curr_norm_tensor.unsqueeze(1), curr_norm_tensor.unsqueeze(0))
                else:
                    if self.before_fusion ==False and self.use_half_feature == True:
                        C,_,_ = bev_feat.shape
                        curr_feat_tensor = bev_feat[:int(C/2), curr_corner_relation[:, 0].long(), curr_corner_relation[:, 1].long()]
                        curr_norm_tensor = torch.norm(curr_feat_tensor, dim=0)
                    else:
                        curr_feat_tensor = bev_feat[:, curr_corner_relation[:, 0].long(), curr_corner_relation[:, 1].long()]
                        curr_norm_tensor = torch.norm(curr_feat_tensor, dim=0)

                    curr_box_cos_sim = torch.mm(curr_feat_tensor.t(), curr_feat_tensor) / torch.mm(
                        curr_norm_tensor.unsqueeze(1), curr_norm_tensor.unsqueeze(0))



                box_total_cos_loss_sum = img_feats[0].new_zeros(9, 9)
                if self.lidar_guided:
                    box_frame_count = img_feats.new_zeros(9, 9)
                else:
                    box_frame_count = 0

                prev_bbox_corners_list = torch.split(bbox_corner_relation, 2, dim=1)
                prev_bbox_corner_mask_list = torch.split(bbox_corner_mask, 1, dim=0)

                for prev_frame_id, (prev_bbox_corner, prev_bbox_corner_mask) in enumerate(zip(prev_bbox_corners_list, prev_bbox_corner_mask_list)):
                # for prev_frame_id, prev_bbox_corner in enumerate(prev_bbox_corners_list):

                    # 历史帧中没有该障碍物 or 障碍物出界了
                    if prev_bbox_corner.max() == 0 or prev_bbox_corner.min() == -1:
                        continue

                    # 当前历史帧没用到，当前帧之前的历史帧也没用到
                    if prev_bev_feat_use[prev_frame_id] == 0:
                        break

                    if self.bilinear_interpolate:
                        prev_bbox_corner = prev_bbox_corner.unsqueeze(0).unsqueeze(0)
                        prev_bev_feat = prev_bev_feat_list[prev_frame_id].unsqueeze(0)
                        _,C,H,W = prev_bev_feat.shape
                        prev_bbox_corner_norm = (2 * prev_bbox_corner / W) - 1
                        bilinear_prev_feat = F.grid_sample(bev_feat,prev_bbox_corner_norm,mode='bilinear')
                        bilinear_prev_feat = bilinear_prev_feat.squeeze(2).squeeze(0)
                        if self.before_fusion == False and self.use_half_feature == True:
                            bilinear_prev_feat = bilinear_prev_feat[:int(C/2),:]

                        prev_norm_tensor = torch.norm(bilinear_prev_feat, dim=0)
                        prev_box_cos_sim = torch.mm(bilinear_prev_feat.t(), bilinear_prev_feat) / torch.mm(
                            prev_norm_tensor.unsqueeze(1), prev_norm_tensor.unsqueeze(0))
                    else:

                        prev_feat_tensor = prev_bev_feat_list[prev_frame_id][:, prev_bbox_corner[:, 0].long(), prev_bbox_corner[:, 1].long()]

                        prev_norm_tensor = torch.norm(prev_feat_tensor, dim=0)

                        prev_box_cos_sim = torch.mm(prev_feat_tensor.t(), prev_feat_tensor) / torch.mm(
                            prev_norm_tensor.unsqueeze(1), prev_norm_tensor.unsqueeze(0))

                    if self.lidar_guided:
                        box_total_cos_loss_sum += prev_bbox_corner_mask.squeeze(0) * abs(curr_box_cos_sim-prev_box_cos_sim)
                        box_frame_count += prev_bbox_corner_mask.squeeze(0)
                    else:
                        box_total_cos_loss_sum += abs(curr_box_cos_sim-prev_box_cos_sim)
                        box_frame_count += 1

                if self.lidar_guided:
                    if box_frame_count.max() > 0:
                            total_cos_loss += box_total_cos_loss_sum/(box_frame_count+1e-6)
                else:
                    if box_frame_count > 0:
                        total_cos_loss += box_total_cos_loss_sum / box_frame_count

        total_cos_loss_sum = self.temporal_consistance_weight*(torch.sum(total_cos_loss)/81/batchsize)

        return {'box_cos_sim_loss': total_cos_loss_sum}
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      gt_bev_high=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        img_feats, pts_feats, depth, bev_feats_list = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)

        if self.depth_loss_type == "v1":
            gt_depth = kwargs['gt_depth']
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
            losses = dict(loss_depth=loss_depth)
        elif self.depth_loss_type == "v2":
            gt_depth = kwargs['gt_depth']
            loss_depth = self.img_view_transformer.get_depth_loss_v2(gt_depth, depth)
            losses = dict(loss_depth=loss_depth)
        elif self.depth_loss_type == "none":
            losses = dict()

        # losses = dict()

        # if hasattr(self, 'bev_structure_reg_head'):
        #     bev_high_feature, bev_high_loss = self.bev_structure_reg_head(bev_feats_list, gt_bev_high)
        #     losses.update(bev_high_loss)
        #     img_feats[0] = torch.cat([img_feats[0], bev_high_feature], dim=1)
        #
        if self.tcl_start_flag:
            corner_relation = kwargs['corner_relation']
            corner_mask = kwargs['corner_mask']

            if self.before_fusion:
                temporal_consistance_loss = self.temporal_consistance_loss(bev_feats_list[0], bev_feats_list, corner_relation,corner_mask)
                # temporal_consistance_loss = self.temporal_consistance_loss(bev_feats_list[0], bev_feats_list, corner_relation)

            else:
                temporal_consistance_loss = self.temporal_consistance_loss(img_feats[0], bev_feats_list, corner_relation,corner_mask)
                # temporal_consistance_loss = self.temporal_consistance_loss(img_feats[0], bev_feats_list, corner_relation)

            losses.update(temporal_consistance_loss)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)


        if self.with_prev:
            sweeps_gt_boxes = kwargs["sweeps_gt_boxes"]
            sweeps_gt_labels = kwargs["sweeps_gt_labels"]
            sweeps_gt_boxes = list(map(list,zip(*sweeps_gt_boxes)))
            sweeps_gt_labels = list(map(list,zip(*sweeps_gt_labels)))

            prev_losses = dict()
            for i in range(len(sweeps_gt_boxes)):
                prev_feat_input = list()
                prev_feat_input.append(bev_feats_list[i+1])
                sweep_losses_pts = self.sweeps_forward_pts_train(prev_feat_input, sweeps_gt_boxes[i],
                                                    sweeps_gt_labels[i], img_metas,
                                                    gt_bboxes_ignore)
                for (key,value) in sweep_losses_pts.items():
                    key = key+"prev"
                    if key in prev_losses.keys():
                        prev_losses[key] = prev_losses[key]
                    else:
                        prev_losses[key] = value

            # for (key, value) in sweep_losses_pts.items():
            #     prev_losses[key] = prev_losses[key]*self.auxiliary_weight

            # sweep_losses_pts = self.sweeps_forward_pts_train(bev_feats_list[1:], sweeps_gt_boxes,
            #                                     sweeps_gt_labels, img_metas,
            #                                     gt_bboxes_ignore)
            # for (key, value) in sweep_losses_pts.items():
            #     prev_losses[key] = sweep_losses_pts[key]*self.auxiliary_weight
            losses.update(prev_losses)

        return losses