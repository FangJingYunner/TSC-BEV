# Copyright (c) OpenMMLab. All rights reserved.
import pickle

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from tools.data_converter import nuscenes_converter as nuscenes_converter


from mmdet3d.core.points import BasePoints, get_points_type
import mmcv


map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def get_gt(ann_infos,info):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in ann_infos:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            raise RuntimeError
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels



def load_points(pts_filename):
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

def get_grid_info(sweep_infos):


    pts_filename = sweep_infos['sweep_info']['data_path']

    file_client = mmcv.FileClient(**{'backend': 'disk'})
    pts_bytes = file_client.get(pts_filename)
    points = np.frombuffer(pts_bytes, dtype=np.float32)

    points = points.reshape(-1, 5)
    # points = points[:, 5]
    attribute_dims = None

    points_class = get_points_type('LIDAR')
    points = points_class(
        points, points_dim=points.shape[-1], attribute_dims=attribute_dims)

    points_lidar = results['points']
    post_rots, post_trans, bda = results['img_inputs'][4:]

    # lidar2ego Coordinate
    lidar2lidarego = np.eye(4, dtype=np.float32)
    lidar2lidarego[:3, :3] = Quaternion(
        results['curr']['lidar2ego_rotation']).rotation_matrix
    lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
    lidar2lidarego = torch.from_numpy(lidar2lidarego)
    points_lidar.tensor[:, :3] = points_lidar.tensor[:, :3].matmul(
        lidar2lidarego[:3, :3].T) + lidar2lidarego[:3, 3].unsqueeze(0)

    points_lidar.rotate(bda)
    points_kept_flag = points_lidar.in_range_3d(self.point_cloud_range)
    points_lidar = points_lidar[points_kept_flag]

    num_points, _ = points_lidar.shape

    coor = ((points_lidar.coord - self.grid_lower_bound.to(points_lidar.device)) /
            self.grid_interval.to(points_lidar.device))
    coor = coor.long().view(num_points, 3)

    # get tensors from the same voxel next to each other
    ranks_bev = coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
    ranks_bev += coor[:, 0] * self.grid_size[0] + coor[:, 1]
    order = ranks_bev.argsort()
    ranks_bev = ranks_bev[order]
    points_lidar = points_lidar[order]
    kept = torch.ones(
        ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
    kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
    interval_starts = torch.where(kept)[0].int()
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]

    kept2 = interval_lengths[:] >= self.grid_minpts  # 保留点数大于等于self.grid_minpts的点
    interval_starts = interval_starts[kept2]
    interval_lengths = interval_lengths[kept2]
    grid_mask_rank = ranks_bev[interval_starts.long()]

    grid_mask = torch.zeros(
        int(self.grid_size[1] * self.grid_size[0]), device=ranks_bev.device, dtype=torch.float)
    grid_mask[grid_mask_rank.long()] = 1.0  # 满足最小点数条件的grid的mask为1

    grid_pts_list = [points_lidar.tensor[start:start + lengths, :3] for start, lengths in
                     zip(interval_starts, interval_lengths)]
    grid_pts_relativehigh = [
        (grid_pts[:, 2].max() - grid_pts[:, 2].min()) / (self.point_cloud_range[5] - self.point_cloud_range[2]) for
        grid_pts in grid_pts_list]
    grid_high = torch.zeros(
        int(self.grid_size[1] * self.grid_size[0]), device=ranks_bev.device, dtype=torch.float)
    grid_high[grid_mask_rank.long()] = torch.tensor(grid_pts_relativehigh)

    gt_grid_mask = grid_mask.view(int(self.grid_size[2]), int(self.grid_size[0]), int(self.grid_size[1]))
    gt_grid_high = grid_high.view(int(self.grid_size[2]), int(self.grid_size[0]), int(self.grid_size[1]))

    results['gt_bev_high'] = (gt_grid_mask, gt_grid_high)


    # results['points'] = points

def get_sweeps_gt(nuscenes, sweeps_infos, curr_anns_instance):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """


    final_sweeps_infos = list()
    for sweep_id, sweep_infos in enumerate(sweeps_infos):
        del_sweep_id = []
        # for sweep_id, sweeps_ann_info in enumerate(sweeps_ann_infos):

        # grid_pts_info = get_grid_info(sweep_infos)

        prev_id = sweep_infos['prev_id']
        prev_sample_ann = sweep_infos['prev_sample']
        sweep_info = sweep_infos['sweep_info']


        ego2global_rotation = sweep_info['ego2global_rotation']
        ego2global_translation = sweep_info['ego2global_translation']
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        gt_boxes = list()
        gt_labels = list()
        curr_gt_boxes_relation = list()
        for ann_token in prev_sample_ann['anns']:
            sample_ann = nuscenes.get('sample_annotation', ann_token)

            if (map_name_from_general_to_detection[sample_ann['category_name']]
                    not in classes
                    or sample_ann['num_lidar_pts'] + sample_ann['num_radar_pts'] <= 0):
                raise RuntimeError

            velocity = nuscenes.box_velocity(sample_ann['token'])
            if np.any(np.isnan(velocity)):
                velocity = np.zeros(3)


            box = Box(
                sample_ann['translation'],
                sample_ann['size'],
                Quaternion(sample_ann['rotation']),
                velocity=velocity,
            )

            box.translate(trans)
            box.rotate(rot)
            box_xyz = np.array(box.center)
            box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
            box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
            box_velo = np.array(box.velocity[:2])
            gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])

            gt_boxes.append(gt_box)
            gt_labels.append(
                classes.index(
                    map_name_from_general_to_detection[sample_ann['category_name']]))

            instance_token = sample_ann['instance_token']
            curr_gt_boxes_relation.append(curr_anns_instance[instance_token])

        final_sweep_info = {}
        final_sweep_info['curr_gt_boxes_relation'] = curr_gt_boxes_relation
        final_sweep_info['prev_id'] = prev_id
        final_sweep_info['gt_boxes'] = gt_boxes
        final_sweep_info['gt_labels'] = gt_labels
        final_sweep_info['timestamp'] = sweep_infos['sweep_info']['timestamp']
        final_sweep_info['data_path'] = sweep_infos['sweep_info']['data_path']

        final_sweeps_infos.append(final_sweep_info)
        #
        # # Use ego coordinate.
        # if (map_name_from_general_to_detection[prev_sample_ann['category_name']]
        #         not in classes
        #         or prev_sample_ann['num_lidar_pts'] + prev_sample_ann['num_radar_pts'] <= 0):
        #
        #     # print(prev_sample_ann['category_name'])
        #     # print('points: {}'.format(prev_sample_ann['num_lidar_pts'] + prev_sample_ann['num_radar_pts']))
        #     # sweeps_anns_infos[ann_id].remove(sweeps_ann_info)
        #     sweeps_anns_infos[ann_id][sweep_id] = []
        #     del_sweep_id.append(sweep_id)
        #     continue
        #
        # ego2global_rotation = prev_sample_data_info['ego2global_rotation']
        # ego2global_translation = prev_sample_data_info['ego2global_translation']
        # trans = -np.array(ego2global_translation)
        # rot = Quaternion(ego2global_rotation).inverse
        #
        # box = Box(
        #     prev_sample_ann['translation'],
        #     prev_sample_ann['size'],
        #     Quaternion(prev_sample_ann['rotation']),
        #     velocity=prev_sample_ann['velocity'],
        # )
        # box.translate(trans)
        # box.rotate(rot)
        # box_xyz = np.array(box.center)
        # box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        # box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        # box_velo = np.array(box.velocity[:2])
        # gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        #
        # gt_label = classes.index(
        #         map_name_from_general_to_detection[prev_sample_ann['category_name']])
        #
        # sweeps_anns_infos[ann_id][sweep_id]['gt_box'] = gt_box
        # sweeps_anns_infos[ann_id][sweep_id]['gt_label'] = gt_label
        # sweeps_anns_infos[ann_id][sweep_id].pop('prev_sample_ann')
        #
        # if len(del_sweep_id) > 0:
        #     all_sweep_id = list(range(len(sweeps_ann_infos)))
        #     keep = list(set(all_sweep_id)-set(del_sweep_id))
        #     # sweeps_anns_infos[ann_id].pop(*del_sweep_id)
        #     sweeps_anns_infos[ann_id] = [sweeps_anns_infos[ann_id][i] for i in keep]

    return final_sweeps_infos


def nuscenes_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)


def add_ann_adj_info_only_key_frame(extra_tag, max_key_frames=10):
    """
    Args:
        only_key_frame (str): pattern to create adjecent frame.
            Optional Parameters： ‘only_key_frame’,'only_sweeps_frame'.'mix_frame'
    """

    nuscenes_version = 'v1.0-mini'
    dataroot = '/media/fjy/HDD2/dataset/nuScenes/nuscenes'
    nuscenes = NuScenes(nuscenes_version, dataroot)
    for set in ['train', 'val']:
        dataset = pickle.load(
            open('/media/fjy/HDD2/dataset/nuScenes/nuscenes/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
        for id in range(len(dataset['infos'])):
            # if id % 10 == 0:
            print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])

            # filter ignored class
            filter_anns = list()
            for ann_id, ann in enumerate(sample['anns']):
                ann_info = nuscenes.get('sample_annotation', ann)
                if (map_name_from_general_to_detection[ann_info['category_name']]
                        not in classes
                        or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
                    continue
                filter_anns.append(ann)
            sample['anns'] = filter_anns #只保留有雷达点或者是我要的类别的标注

            ann_infos = list()
            curr_anns_instance = dict()
            for ann_id, ann in enumerate(sample['anns']):
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)#给每个ann增加速度标注
                curr_anns_instance[ann_info['instance_token']] = ann_id#给每个ann一个数字名称

            #get key sweeps and filter anns that not in current frame
            key_sweeps_info = []
            for prev_id, sweep_info in enumerate(info['sweeps']):
                prev_sample_data = nuscenes.get('sample_data', sweep_info['sample_data_token'])

                if prev_sample_data['is_key_frame'] == True and len(key_sweeps_info) < max_key_frames:
                    prev_sample = nuscenes.get('sample', prev_sample_data['sample_token'])

                    prev_ann_token = []
                    for prev_sample_ann_token in prev_sample['anns']:
                        prev_sample_ann = nuscenes.get('sample_annotation', prev_sample_ann_token)
                        #如果current frame的ann同样存在于sweeps帧中就保留下来
                        if prev_sample_ann['instance_token'] in curr_anns_instance.keys():
                            prev_ann_token.append(prev_sample_ann_token)

                    prev_sample['anns'] = prev_ann_token

                    key_sweeps_info.append({'prev_id': prev_id,
                                            'sweep_info': sweep_info,
                                            'prev_sample': prev_sample})

            dataset['infos'][id]['ann_infos'] = get_gt(ann_infos, dataset['infos'][id])
            dataset['infos'][id]['sweeps'] = get_sweeps_gt(nuscenes, key_sweeps_info, curr_anns_instance)
            dataset['infos'][id]['scene_token'] = sample['scene_token']

        with open('/media/fjy/HDD2/dataset/nuScenes/nuscenes/%s_infos_%s.pkl' % (extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)


if __name__ == '__main__':
    dataset = 'nuscenes'
    version = 'v1.0'
    train_version = f'{version}-mini'
    # root_path = './data/nuscenes'
    root_path = '/media/fjy/HDD2/dataset/nuScenes/nuscenes'

    extra_tag = 'bevdetv2-nuscenes-miniTC3'
    nuscenes_data_prep(
        root_path=root_path,
        info_prefix=extra_tag,
        version=train_version,
        max_sweeps=110)

    print('add_ann_infos')
    add_ann_adj_info_only_key_frame(extra_tag, max_key_frames=10)
