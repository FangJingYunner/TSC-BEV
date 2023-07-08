# Copyright (c) OpenMMLab. All rights reserved.
import time

import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int, optional): Number of sweeps. Defaults to 10.
        load_dim (int, optional): Dimension number of the loaded points.
            Defaults to 5.
        use_dim (list[int], optional): Which dimension to use.
            Defaults to [0, 1, 2, 4].
        time_dim (int, optional): Which dimension to represent the timestamps
            of each points. Defaults to 4.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool, optional): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool, optional): Whether to remove close points.
            Defaults to False.
        test_mode (bool, optional): If `test_mode=True`, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 time_dim=4,
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.time_dim = time_dim
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        assert time_dim < load_dim, \
            f'Expect the timestamp dimension < {load_dim}, got {time_dim}'
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float, optional): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, self.time_dim] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, self.time_dim] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        adjacent_points_list = []
        for adj_index,adj in enumerate(results['adjacent']):
            pts_filename = adj['lidar_path']
            points = self._load_points(pts_filename)
            points = points.reshape(-1, self.load_dim)
            points = points[:, self.use_dim]
            attribute_dims = None

            if self.shift_height:
                floor_height = np.percentile(points[:, 2], 0.99)
                height = points[:, 2] - floor_height
                points = np.concatenate(
                    [points[:, :3],
                     np.expand_dims(height, 1), points[:, 3:]], 1)
                attribute_dims = dict(height=3)

            if self.use_color:
                assert len(self.use_dim) >= 6
                if attribute_dims is None:
                    attribute_dims = dict()
                attribute_dims.update(
                    dict(color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]))

            points_class = get_points_type(self.coord_type)
            points = points_class(
                points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
            # sweep['points'] = points
            # results['adjacent'][adj_index]['points'] =points
            adjacent_points_list.append(points)
            # sweep_points_list.append(points)
            # time.sleep(0.003)
        results['adjacent_points_list']=adjacent_points_list

        return results


    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromDict(LoadPointsFromFile):
    """Load Points From Dict."""

    def __call__(self, results):
        assert 'points' in results
        return results


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype=np.int64,
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int64)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str


@PIPELINES.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()#返回从小到大索引
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])#如果两个点在一个像素中，就取近处的
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results


def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class PrepareImageInputs(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        ego_cam='CAM_FRONT',
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.ego_cam = ego_cam

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor2ego_transformation(self,
                                      cam_info,
                                      key_info,
                                      cam_name,
                                      ego_cam=None):
        if ego_cam is None:
            ego_cam = cam_name
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sweepsensor2sweepego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepsensor2sweepego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
        sweepsensor2sweepego[3, 3] = 1
        sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
        sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        sweepego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        sweepego2global = sweepego2global_rot.new_zeros((4, 4))
        sweepego2global[3, 3] = 1
        sweepego2global[:3, :3] = sweepego2global_rot
        sweepego2global[:3, -1] = sweepego2global_tran

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][ego_cam]['ego2global_rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info['cams'][ego_cam]['ego2global_translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        sweepsensor2keyego = \
            global2keyego @ sweepego2global @ sweepsensor2sweepego

        # global sensor to cur ego
        # w, x, y, z = key_info['cams'][cam_name]['ego2global_rotation']
        # keyego2global_rot = torch.Tensor(
        #     Quaternion(w, x, y, z).rotation_matrix)
        # keyego2global_tran = torch.Tensor(
        #     key_info['cams'][cam_name]['ego2global_translation'])
        # keyego2global = keyego2global_rot.new_zeros((4, 4))
        # keyego2global[3, 3] = 1
        # keyego2global[:3, :3] = keyego2global_rot
        # keyego2global[:3, -1] = keyego2global_tran
        # global2keyego = keyego2global.inverse()

        # cur ego to sensor
        w, x, y, z = key_info['cams'][cam_name]['sensor2ego_rotation']
        keysensor2keyego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keysensor2keyego_tran = torch.Tensor(
            key_info['cams'][cam_name]['sensor2ego_translation'])
        keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
        keysensor2keyego[3, 3] = 1
        keysensor2keyego[:3, :3] = keysensor2keyego_rot
        keysensor2keyego[:3, -1] = keysensor2keyego_tran
        keyego2keysensor = keysensor2keyego.inverse()
        keysensor2sweepsensor = (
            keyego2keysensor @ global2keyego @ sweepego2global
            @ sweepsensor2sweepego).inverse()
        return sweepsensor2keyego, keysensor2sweepsensor

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        sensor2sensors = []
        for cam_name in cam_names:#每个摄像头都有自己的增强
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2keyego, sensor2sensor = \
                self.get_sensor2ego_transformation(results['curr'],
                                                   results['curr'],
                                                   cam_name,
                                                   self.ego_cam)
            rot = sensor2keyego[:3, :3]
            tran = sensor2keyego[:3, 3]
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            sensor2sensors.append(sensor2sensor)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                trans_adj = []
                rots_adj = []
                sensor2sensors_adj = []
                for cam_name in cam_names:
                    adjsensor2keyego, sensor2sensor = \
                        self.get_sensor2ego_transformation(adj_info,
                                                           results['curr'],
                                                           cam_name,
                                                           self.ego_cam)
                    rot = adjsensor2keyego[:3, :3]
                    tran = adjsensor2keyego[:3, 3]
                    rots_adj.append(rot)
                    trans_adj.append(tran)
                    sensor2sensors_adj.append(sensor2sensor)
                rots.extend(rots_adj)#从sweep sensor 到key ego 的变换
                trans.extend(trans_adj)#从sweep sensor 到key ego 的变换
                sensor2sensors.extend(sensor2sensors_adj) # keysensor2sweepsensor
        imgs = torch.stack(imgs)

        rots = torch.stack(rots)#从sweep sensor 到key ego 的变换
        trans = torch.stack(trans)#从sweep sensor 到key ego 平移
        intrins = torch.stack(intrins)#相机内参，key 和sweep都一致
        post_rots = torch.stack(post_rots)#图像数据增强
        post_trans = torch.stack(post_trans)#图像数据增强
        sensor2sensors = torch.stack(sensor2sensors)# keysensor2sweepsensor
        results['canvas'] = canvas
        results['sensor2sensors'] = sensor2sensors
        return (imgs, rots, trans, intrins, post_rots, post_trans)

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadAnnotationsBEVDepth(object):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)#中心点旋转
            gt_boxes[:, 3:6] *= scale_ratio#whl缩放
            gt_boxes[:, 6] += rotate_angle#朝向角度
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']
        gt_boxes, gt_labels = torch.Tensor(np.array(gt_boxes)), torch.tensor(np.array(gt_labels))
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        #origin
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot

        #change
        # gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
        #                                        flip_dx, flip_dy)
        # bda_mat[:3, :3] = bda_mat
        # bda_rot = bda_mat[:3, :3]

        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
                                 post_trans, bda_rot)

        results['bev_pts_aug'] = (rotate_bda, scale_bda, flip_dx, flip_dy)
        return results


@PIPELINES.register_module()
class LoadSweepsAnnotationsBEVDepth(object):

    def __init__(self, bda_aug_conf, classes, point_cloud_range, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)#中心点旋转
            gt_boxes[:, 3:6] *= scale_ratio#whl缩放
            gt_boxes[:, 6] += rotate_angle#朝向角度
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):

        (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot) = results['img_inputs']
        (rotate_bda, scale_bda, flip_dx, flip_dy) = results['bev_pts_aug']

        results["sweeps_gt_boxes"] = list()
        results["sweeps_gt_labels"] = list()
        results["adjego2currego"] = list()

        # N, _, H, W = imgs.shape
        N = len(results["adjacent"])+1
        extra = [
            rots.view(N, 6, 3, 3),
            trans.view(N, 6, 3),
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans = extra

        # n, c, h, w = imgs.shape
        # _, v, _ = trans[0].shape
        # get transformation from current ego frame to adjacent ego frame
        # transformation from current camera frame to current ego frame
        c02l0 = torch.zeros((4, 4), dtype=rots[0].dtype).to(rots[0])
        c02l0[:3, :3] = rots[0][0]
        c02l0[:3, 3] = trans[0][0]
        c02l0[3, 3] = 1


        # add bev data augmentation
        # bda_ = torch.zeros((4, 4), dtype=rots[0].dtype).to(rots[0])
        # bda_[:3, :3] = bda_rot
        # bda_[3, 3] = 1
        # c02l0 = bda_.matmul(c02l0)


        for ind, sweep_ann_infos in enumerate(results['adjacent']):
            # transformation from adjacent camera frame to current ego frame
            c12l0 = torch.zeros((4, 4), dtype=rots[0].dtype).to(rots[0])
            c12l0[:3, :3] = rots[0][ind+1]
            c12l0[:3, 3] = trans[0][ind+1]
            c12l0[3, 3] = 1
            # c12l0 = bda_.matmul(c12l0)
            # transformation from current ego frame to adjacent ego frame
            l02l1 = c02l0.matmul(torch.inverse(c12l0))
            l12l0 = torch.inverse(l02l1)
            '''
              c02l0 * inv(c12l0)
            = c02l0 * inv(l12l0 * c12l1)
            = c02l0 * inv(c12l1) * inv(l12l0)
            = l02l1 # c02l0==c12l1
            '''


            sweep_gt_boxes, sweep_gt_labels = sweep_ann_infos['ann_infos']
            sweep_gt_boxes, sweep_gt_labels = torch.Tensor(np.array(sweep_gt_boxes)), torch.tensor(np.array(sweep_gt_labels))

            if sweep_gt_boxes.shape[0] > 0:
                # sweep_gt_boxes[:,0:3] = (l02l1[:3,:3] @ sweep_gt_boxes[:,0:3].unsqueeze(-1) + l02l1[:3,3].unsqueeze(-1)).squeeze(-1)
                # sweep_gt_boxes_t = LiDARInstance3DBoxes(sweep_gt_boxes, box_dim=sweep_gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5))
                # sweep_gt_boxes_t = sweep_gt_boxes_t.rotate(l02l1[:3, :3])
                # sweep_gt_boxes_t.rotate(bda_rot)
                # try:
                #     if len(sweep_gt_boxes) > 0:
                #gt sweep bbox adjego convert to current ego
                rot_sin = (l12l0[0, 1]+l12l0[1,0])/2
                rot_cos = (l12l0[0, 0]+l12l0[1,1])/2
                angle = np.arctan2(rot_sin, rot_cos)
                sweep_gt_boxes[:, :3] = sweep_gt_boxes[:,:3].matmul(l12l0[:3, :3].T)+l12l0[:3, 3].unsqueeze(0)
                sweep_gt_boxes[:, 6] += angle
                # rotate velo vector
                sweep_gt_boxes[:, 7:9] = sweep_gt_boxes[:, 7:9].matmul(l12l0[:2, :2].T)
                # except:
                #     print("sweep_gt_boxes shape:{}".format(sweep_gt_boxes.shape))
                #     print("l12l0 shape:{}".format(l12l0.shape))

                #bda
                sweep_gt_boxes, bda_rot = self.bev_transform(sweep_gt_boxes, rotate_bda, scale_bda,flip_dx, flip_dy)

                sweep_gt_boxes = LiDARInstance3DBoxes(sweep_gt_boxes, box_dim=sweep_gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5))

                bev_range = self.pcd_range[[0, 1, 3, 4]]

                mask = sweep_gt_boxes.in_range_bev(bev_range)
                sweep_gt_boxes = sweep_gt_boxes[mask]
                # mask is a torch tensor but gt_labels_3d is still numpy array
                # using mask to index gt_labels_3d will cause bug when
                # len(gt_labels_3d) == 1, where mask=1 will be interpreted
                # as gt_labels_3d[1] and cause out of index error
                sweep_gt_labels = sweep_gt_labels[mask.numpy().astype(np.bool)]

                # limit rad to [-pi, pi]
                sweep_gt_boxes.limit_yaw(offset=0.5, period=2 * np.pi)

            results["sweeps_gt_boxes"].append(sweep_gt_boxes)
            results["sweeps_gt_labels"].append(sweep_gt_labels)
            results["adjego2currego"].append(l12l0)

        return results

@PIPELINES.register_module()
class PointToGridStructure(object):

    def __init__(self, grid_config, point_cloud_range,grid_minpts=5):
        self.grid_minpts = grid_minpts
        self.grid_config = grid_config
        self.create_grid_infos(**grid_config)
        self.point_cloud_range = point_cloud_range

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])

    def __call__(self, results):
        points_lidar = results['points']
        post_rots, post_trans, bda = results['img_inputs'][4:]

        #lidar2ego Coordinate
        lidar2lidarego = np.eye(4, dtype=np.float32)
        lidar2lidarego[:3, :3] = Quaternion(
            results['curr']['lidar2ego_rotation']).rotation_matrix
        lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
        lidar2lidarego = torch.from_numpy(lidar2lidarego)
        points_lidar.tensor[:, :3] = points_lidar.tensor[:, :3].matmul(
            lidar2lidarego[:3, :3].T) + lidar2lidarego[:3, 3].unsqueeze(0)

        points_lidar.tensor[:, :3] = points_lidar.tensor[:, :3].matmul(bda.T)

        points_kept_flag = points_lidar.in_range_3d(self.point_cloud_range)
        points_lidar = points_lidar[points_kept_flag]

        num_points, _ = points_lidar.shape

        coor = ((points_lidar.coord - self.grid_lower_bound.to(points_lidar.device)) /
                self.grid_interval.to(points_lidar.device))
        coor = coor.long().view(num_points, 3)

        # get tensors from the same voxel next to each other
        # ranks_bev = coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev = coor[:, 0] * self.grid_size[0] + coor[:, 1]
        order = ranks_bev.argsort()
        ranks_bev = ranks_bev[order]
        # points_lidar = points_lidar[order]
        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]

        kept2 = interval_lengths[:] >= self.grid_minpts#保留点数大于等于self.grid_minpts的点
        interval_starts = interval_starts[kept2]
        # interval_lengths = interval_lengths[kept2]
        grid_mask_rank = ranks_bev[interval_starts.long()]

        grid_mask = torch.zeros(
            int(self.grid_size[1] * self.grid_size[0]), device=ranks_bev.device, dtype=torch.float)
        grid_mask[grid_mask_rank.long()] = 1.0  # 满足最小点数条件的grid的mask为1

        # try:
        #     grid_mask[grid_mask_rank.long()] = 1.0  # 满足最小点数条件的grid的mask为1
        # except:
        #     aa = 2

        # grid_pts_list = [points_lidar.tensor[start:start + lengths, :3] for start, lengths in
        #                  zip(interval_starts, interval_lengths)]
        # grid_pts_relativehigh = [
        #     (grid_pts[:, 2].max() - grid_pts[:, 2].min()) / (self.point_cloud_range[5] - self.point_cloud_range[2]) for
        #     grid_pts in grid_pts_list]
        # grid_high = torch.zeros(
        #     int(self.grid_size[1] * self.grid_size[0]), device=ranks_bev.device, dtype=torch.float)
        # grid_high[grid_mask_rank.long()] = torch.tensor(grid_pts_relativehigh)

        gt_grid_mask = grid_mask.view(int(self.grid_size[2]), int(self.grid_size[0]), int(self.grid_size[1]))
        # gt_grid_high = grid_high.view(int(self.grid_size[2]), int(self.grid_size[0]), int(self.grid_size[1]))

        # results['gt_bev_high'] = (gt_grid_mask, gt_grid_high)
        results['gt_bev_mask'] = gt_grid_mask
        # results['gt_bev_high'] = gt_grid_high

        adjacent_gt_bev_masks = []
        for index, adj in enumerate(results['adjacent']):
            # if index < len(results['sweeps']):
            # assert adj['timestamp'] == results['sweeps'][index]['timestamp']
            points_lidar = results['adjacent_points_list'][index]
            adj2current = results['adjego2currego'][index]

            _, _, bda = results['img_inputs'][4:]

            # lidar2ego Coordinate
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                adj['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = adj['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)
            points_lidar.tensor[:, :3] = points_lidar.tensor[:, :3].matmul(
                lidar2lidarego[:3, :3].T) + lidar2lidarego[:3, 3].unsqueeze(0)

            # points_lidar = points_lidars.clone()



            #pts convert to current frame
            # points_lidar.tensor[:,:3] = points_lidar.tensor[:,:3].matmul(adj2current[:3, :3].T) +adj2current[:3, 3].unsqueeze(0)
            # points_lidar.rotate(bda)
            points_lidar.tensor[:, :3] = points_lidar.tensor[:, :3].matmul(bda.T)

            # points_lidar.tensor[:, :3].matmul(
            #     lidar2lidarego[:3, :3].T) + lidar2lidarego[:3, 3].unsqueeze(0)

            points_kept_flag = points_lidar.in_range_3d(self.point_cloud_range)
            points_lidar = points_lidar[points_kept_flag]

            num_points, _ = points_lidar.shape

            coor = ((points_lidar.coord - self.grid_lower_bound.to(points_lidar.device)) /
                    self.grid_interval.to(points_lidar.device))
            coor = coor.long().view(num_points, 3)

            # get tensors from the same voxel next to each other
            # ranks_bev = coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
            ranks_bev = coor[:, 0] * self.grid_size[0] + coor[:, 1]
            order = ranks_bev.argsort()
            ranks_bev = ranks_bev[order]
            # points_lidar = points_lidar[order]
            kept = torch.ones(
                ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
            kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
            interval_starts = torch.where(kept)[0].int()
            interval_lengths = torch.zeros_like(interval_starts)
            interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
            interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]

            kept2 = interval_lengths[:] >= 1  # 保留点数大于等于self.grid_minpts的点
            interval_starts = interval_starts[kept2]
            # interval_lengths = interval_lengths[kept2]
            grid_mask_rank = ranks_bev[interval_starts.long()]

            grid_mask = torch.zeros(
                int(self.grid_size[1] * self.grid_size[0]), device=ranks_bev.device, dtype=torch.float)
            grid_mask[grid_mask_rank.long()] = 1.0  # 满足最小点数条件的grid的mask为1

            # try:
            #     grid_mask[grid_mask_rank.long()] = 1.0  # 满足最小点数条件的grid的mask为1
            # except:
            #     aa = 2


            # grid_pts_list = [points_lidar.tensor[start:start + lengths, :3] for start, lengths in
            #                  zip(interval_starts, interval_lengths)]
            # grid_pts_relativehigh = [
            #     (grid_pts[:, 2].max() - grid_pts[:, 2].min()) / (
            #                 self.point_cloud_range[5] - self.point_cloud_range[2]) for
            #     grid_pts in grid_pts_list]
            # grid_high = torch.zeros(
            #     int(self.grid_size[1] * self.grid_size[0]), device=ranks_bev.device, dtype=torch.float)
            # grid_high[grid_mask_rank.long()] = torch.tensor(grid_pts_relativehigh)

            gt_grid_mask = grid_mask.view(int(self.grid_size[2]), int(self.grid_size[0]), int(self.grid_size[1]))
            # gt_grid_high = grid_high.view(int(self.grid_size[2]), int(self.grid_size[0]), int(self.grid_size[1]))

            # results['sweeps'][index]['gt_bev_high'] = (gt_grid_mask, gt_grid_high)
            # results['adjacent'][index]['gt_bev_mask'] = gt_grid_mask
            # results['adjacent'][index]['gt_bev_high'] = gt_grid_high
            adjacent_gt_bev_masks.append(gt_grid_mask)

        adjacent_gt_bev_masks = torch.stack(adjacent_gt_bev_masks).squeeze(1)
        results['adjacent_gt_bev_masks'] = adjacent_gt_bev_masks

        return results

@PIPELINES.register_module()
class BEVObjectCornerAnnotation(object):

    def __init__(self, grid_config, point_cloud_range, lidar_guided=False, grid_minpts=5):
        self.grid_minpts = grid_minpts
        self.grid_config = grid_config
        self.create_grid_infos(**grid_config)
        self.point_cloud_range = point_cloud_range
        self.lidar_guided = lidar_guided
    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)#中心点旋转
            gt_boxes[:, 3:6] *= scale_ratio#whl缩放
            gt_boxes[:, 6] += rotate_angle#朝向角度
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])

    def __call__(self, results):

        gt_bbox_mask = results['gt_bbox_rangemask']
        curr_gt_bboxes_3d = results['gt_bboxes_3d']
        curr_gt_labels_3d = results['gt_labels_3d']

        gt_bev_mask = results['gt_bev_mask']
        # gt_bev_high = results['gt_bev_high']

        raw_index = list(range(len(gt_bbox_mask)))
        gt_bbox_index = [index for index in raw_index if gt_bbox_mask[index] == True]

        # (_, rots, trans, intrins, post_rots,
        #  post_trans, bda_rot) = results['img_inputs']

        curr_gt_bboxes_3d = curr_gt_bboxes_3d.corners[...,:2]
        (curr_corner_x6y6, x0y0, x0y1, curr_corner_x8y8, curr_corner_x0y0, x1y0, x1y1, curr_corner_x2y2) = torch.split(curr_gt_bboxes_3d, 1, dim=1)
        # use 9 corner discript one object
        # x0y0z0    x1y1z1    x2y2z2
        # x3y3z3    x4y4z4    x5y5z5
        # x6y6z6    x7y7z7    x8y8z8
        curr_corner_x1y1 = (curr_corner_x0y0 + curr_corner_x2y2) / 2
        curr_corner_x3y3 = (curr_corner_x0y0 + curr_corner_x6y6) / 2
        curr_corner_x7y7 = (curr_corner_x8y8 + curr_corner_x6y6) / 2
        curr_corner_x5y5 = (curr_corner_x8y8 + curr_corner_x2y2) / 2
        curr_corner_x4y4 = (curr_corner_x3y3 + curr_corner_x5y5) / 2

        curr_corner_points = torch.cat(
            [curr_corner_x0y0, curr_corner_x1y1, curr_corner_x2y2, curr_corner_x3y3, curr_corner_x4y4,
             curr_corner_x5y5, curr_corner_x6y6, curr_corner_x7y7, curr_corner_x8y8], dim=1)

        curr_corner_coor = ((curr_corner_points - self.grid_lower_bound[:2].to(curr_corner_points.device)) /
                self.grid_interval[:2].to(curr_corner_points.device))
        curr_corner_coor = curr_corner_coor

        valid_corner_mask = ((curr_corner_coor[:, :, 0] < 0) |
                             (curr_corner_coor[:, :, 0] >= self.grid_size[0]) |
                             (curr_corner_coor[:, :, 1] < 0) |
                             (curr_corner_coor[:, :, 1] >= self.grid_size[1]))
        valid_corner_mask = valid_corner_mask.unsqueeze(dim=-1).expand_as(curr_corner_coor)

        curr_corner_coor[valid_corner_mask] = -1

        curr_corner_tensor = torch.zeros(len(gt_bbox_index), 9, 2 * (len(results['adjacent']) + 1), dtype=torch.float)

        curr_corner_mask = torch.zeros((len(gt_bbox_index), 1 * (len(results['adjacent']) + 1),9,9), dtype=torch.long)
        curr_corner_tensor[:, :, :2] = curr_corner_coor

        if self.lidar_guided:
            # 将gt_bev_mask中x和y坐标的索引是否大于0的信息添加到curr_corner_coor的第三个维度中
            mask = torch.tensor(
                [gt_bev_mask[..., curr_corner_coor[i, j, 0].long(), curr_corner_coor[i, j, 1].long()] for i in
                 range(curr_corner_coor.shape[0]) for j in range(curr_corner_coor.shape[1])], dtype=torch.float32)
            mask = mask.reshape(curr_corner_coor.shape[0], curr_corner_coor.shape[1], 1)
            # curr_corner_coor = torch.cat((curr_corner_coor[..., :2], mask), dim=-1).long()
            # nonzero_idx = torch.nonzero(curr_corner_coor_t[:,:,-1])

            # row_indices, col_indices, depth_indices = torch.where(mask == 1)
            # curr_corner_mask[:, depth_indices, row_indices, row_indices[:, None]] = 1
            for i,obj_mask in enumerate(mask):
                row_indices, col_indices = torch.where(obj_mask == 1)
                curr_corner_mask[i, 0, row_indices, row_indices[:, None]] = 1

        # import matplotlib.pyplot as plt
        # plt.imshow(gt_bev_mask.squeeze(0).numpy(), cmap='binary')
        # plt.show()
        # gt_bev_mask_b = torch.zeros_like(gt_bev_mask)
        # gt_bev_mask_b = gt_bev_mask.clone()
        # for i in range(curr_corner_coor.shape[0]):
        #     for j in range(curr_corner_coor.shape[1]):
        #         gt_bev_mask_b[:,curr_corner_coor[i, j, 0].long(),curr_corner_coor[i, j, 1].long()] = 2
        # plt.imshow(gt_bev_mask_b.squeeze(0).numpy(), cmap='gray')
        # plt.show()
        #
        # gt_bev_mask_b = gt_bev_mask.clone()
        # for i in range(curr_corner_coor.shape[0]):
        #     for j in range(curr_corner_coor.shape[1]):
        #         if curr_corner_mask[i,0, j, 0] == 1:
        #             gt_bev_mask_b[:,curr_corner_coor[i, j, 0].long(),curr_corner_coor[i, j, 1].long()] = 2
        # plt.imshow(gt_bev_mask_b.squeeze(0).numpy(), cmap='gray')
        # plt.show()
        #
        # gt_bev_mask_b = gt_bev_mask.clone()
        # for i in range(curr_corner_coor.shape[0]):
        #     for j in range(curr_corner_coor.shape[1]):
        #         if curr_corner_mask[i,0, j, 0] == 0:
        #             gt_bev_mask_b[:,curr_corner_coor[i, j, 0].long(),curr_corner_coor[i, j, 1].long()] = 3
        # plt.imshow(gt_bev_mask_b.squeeze(0).numpy(), cmap='gray')
        # plt.show()


        (rotate_bda, scale_bda, flip_dx, flip_dy) = results['bev_pts_aug']
        for index, adj in enumerate(results['adjacent']):
            # filter bbox not in curr frame bbox
            if 'relation2curr' in adj.keys():
                sweeps_boxes_relation = adj['relation2curr']['curr_gt_boxes_relation']
                sweeps_gt_bboxes = adj['relation2curr']['gt_boxes']
                sweeps_gt_labels = adj['relation2curr']['gt_labels']
                gt_bev_mask = results["adjacent_gt_bev_masks"][index].unsqueeze(0)
                adjego2currego = results['adjego2currego'][index]

                # sweeps_gt_bev_high = adj['gt_bev_high']

                # sweep_keep_obj_id = set(gt_bbox_index) & set(sweeps_boxes_relation)
                # filtered_gt_boxes = list()
                # filtered_gt_labels = list()
                # for i ,id in enumerate(sweeps_boxes_relation):
                #     if id in sweep_keep_obj_id:
                #         filtered_gt_boxes.(sweeps_gt_bboxes[i])
                #         filtered_gt_labels.append(sweeps_gt_labels[i])

                # sweep_keep_obj_id = set(gt_bbox_index) & set(sweeps_boxes_relation)
                # keep_indices = np.isin(sweeps_boxes_relation, list(sweep_keep_obj_id))
                # filtered_gt_boxes = sweeps_gt_bboxes[keep_indices]
                # filtered_gt_labels = sweeps_gt_labels[keep_indices]

                sweep_keep_obj_id = set(gt_bbox_index) & set(sweeps_boxes_relation)
                keep_indices = np.where(np.isin(sweeps_boxes_relation,list(sweep_keep_obj_id)))
                filtered_gt_boxes = [sweeps_gt_bboxes[i] for i in keep_indices[0]]
                filtered_gt_labels = [sweeps_gt_labels[i] for i in keep_indices[0]]

                # results['sweeps'][index]['curr_gt_boxes_relation'] = sweep_keep_obj_id
                # results['sweeps'][index]['gt_bboxes'] = filtered_gt_bbox
                # results['sweeps'][index]['gt_labels'] = filtered_gt_label

                if len(filtered_gt_boxes) == 0:
                    # print("len(filtered_gt_boxes: {}".format(len(filtered_gt_boxes)))
                    continue
                # pass
                # # gt_boxes, gt_labels = results['ann_infos']
                filtered_gt_boxes, filtered_gt_labels = torch.Tensor(np.array(filtered_gt_boxes)), torch.tensor(np.array(filtered_gt_labels))


                # rot_sin = (adjego2currego[0, 1]+adjego2currego[1, 0])/2
                # rot_cos = (adjego2currego[0, 0]+adjego2currego[1, 1])/2
                # angle = np.arctan2(rot_sin, rot_cos)
                # filtered_gt_boxes[:, 0:3] = filtered_gt_boxes[:, 0:3].matmul(adjego2currego[:3, :3].T)+adjego2currego[:3, 3].unsqueeze(0)
                # filtered_gt_boxes[:, 6] += angle
                # # rotate velo vector
                # filtered_gt_boxes[:, 7:9] = filtered_gt_boxes[:, 7:9].matmul(adjego2currego[:2, :2].T)

                #bda
                filtered_gt_boxes, bda_rot = self.bev_transform(filtered_gt_boxes, rotate_bda, scale_bda, flip_dx, flip_dy)


                filtered_gt_boxes = LiDARInstance3DBoxes(filtered_gt_boxes, box_dim=filtered_gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5))

                # filtered_gt_boxes.rotate(bda_rot)

                (corner_x6y6, x0y0, x0y1, corner_x8y8, corner_x0y0, x1y0, x1y1, corner_x2y2) = torch.split(
                    filtered_gt_boxes.corners[..., :2], 1, dim=1)

                #use 9 corner discript one object
                # x0y0z0    x1y1z1    x2y2z2
                # x3y3z3    x4y4z4    x5y5z5
                # x6y6z6    x7y7z7    x8y8z8
                corner_x1y1 = (corner_x0y0+corner_x2y2)/2
                corner_x3y3 = (corner_x0y0+corner_x6y6)/2
                corner_x7y7 = (corner_x8y8+corner_x6y6)/2
                corner_x5y5 = (corner_x8y8+corner_x2y2)/2
                corner_x4y4 = (corner_x3y3+corner_x5y5)/2

                corner_points = torch.cat(
                    [corner_x0y0, corner_x1y1, corner_x2y2, corner_x3y3, corner_x4y4, corner_x5y5,
                     corner_x6y6, corner_x7y7, corner_x8y8], dim=1)
                # corner_points = corner_points.reshape(N*C,D)

                # grid_mask = torch.zeros(
                #     int(self.grid_size[1] * self.grid_size[0]), device=corner_points.device, dtype=torch.float)
                # grid_mask = grid_mask.reshape(int(self.grid_size[0]), int(self.grid_size[1]),int(self.grid_size[2]))
                # grid_mask = torch.expand(int(self.grid_size[0]), int(self.grid_size[1]),)
                corner_points_coors = ((corner_points - self.grid_lower_bound[:2].to(corner_points.device)) /
                        self.grid_interval[:2].to(corner_points.device))

                valid_corner_mask = ((corner_points_coors[:, :, 0] < 0) |
                                     (corner_points_coors[:, :, 0] >= self.grid_size[0]) |
                                     (corner_points_coors[:, :, 1] < 0) |
                                     (corner_points_coors[:, :, 1] >= self.grid_size[1]))
                valid_corner_mask = valid_corner_mask.unsqueeze(dim=-1).expand_as(corner_points_coors)

                corner_points_coors[valid_corner_mask] = 0

                # sweep_keep_obj_id = set(gt_bbox_index) & set(sweeps_boxes_relation)
                keep_indices = np.where(np.isin(gt_bbox_index, list(sweep_keep_obj_id)))
                keep_indices = np.array(keep_indices).squeeze()
                curr_corner_tensor[keep_indices, :, 2*(1+index):2*(2+index)] = corner_points_coors

                # if self.lidar_guided:
                #     mask = torch.tensor(
                #         [gt_bev_mask[..., corner_points_coors[i, j, 0].long(), corner_points_coors[i, j, 1].long()] for i in
                #          range(corner_points_coors.shape[0]) for j in range(corner_points_coors.shape[1])], dtype=torch.float32)
                #     mask = mask.reshape(corner_points_coors.shape[0], corner_points_coors.shape[1], 1)
                #
                #     for i, obj_mask in enumerate(mask):
                #         row_indices, col_indices = torch.where(obj_mask == 1)
                #         curr_corner_mask[i, index+1, row_indices, row_indices[:, None]] = 1

                # plt.imshow(gt_bev_mask.squeeze(0).numpy(), cmap='binary')
                # plt.show()
                # # gt_bev_mask_b = torch.zeros_like(gt_bev_mask)


        results['corner_relation'] = curr_corner_tensor
        results['corner_mask'] = curr_corner_mask

        return results

