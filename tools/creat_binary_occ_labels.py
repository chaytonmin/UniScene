# some codes from https://github.com/FANG-MING/occupancy-for-nuscenes
import os
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
from open3d import *
#from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.geometry_utils import points_in_box
import os.path as osp
from functools import partial
from points_process import *
from sklearn.neighbors import KDTree
import open3d as o3d
import argparse
INTER_STATIC_POINTS = {}
INTER_STATIC_POSE = {}
INTER_STATIC_LABEL = {}

def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--dataroot',
        type=str,
        default='./project/data/nuscenes/',
        help='specify the root path of dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./project/data/nuscenes/occ-3d/',
        required=False,
        help='specify sweeps of lidar per example')
    parser.add_argument(
        '--num_sweeps',
        type=int,
        default=2,
        required=False,
        help='specify sweeps of lidar per example')
    args = parser.parse_args()
    return args


def get_frame_info(frame, nusc: NuScenes, gt_from='lidarseg'):
    '''
    get frame info
    return: frame_info (Dict):

    '''
    sd_rec = nusc.get('sample_data', frame['data']['LIDAR_TOP'])
    lidar_path, boxes, _ = nusc.get_sample_data(frame['data']['LIDAR_TOP'])

    pc = LidarPointCloud.from_file(nusc.dataroot+sd_rec['filename']) 

    cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    instance_tokens = [nusc.get('sample_annotation', token)['instance_token'] for token in frame['anns']]
    frame_info = {
        'pc': pc,
        'token': frame['token'],
        'lidar_token': frame['data']['LIDAR_TOP'],
        'cs_record': cs_record,
        'pose_record': pose_record,
        'boxes': boxes,
        'anno_token': frame['anns'],
        'instance_tokens': instance_tokens,
        'timestamp': frame['timestamp'],
    }
    return frame_info


def get_intermediate_frame_info(nusc: NuScenes, prev_frame_info, lidar_rec, flag):
    intermediate_frame_info = dict()
    pc = LidarPointCloud.from_file(nusc.dataroot+lidar_rec['filename']) 
    intermediate_frame_info['pc'] = pc
    intermediate_frame_info['pc'].points = remove_close(intermediate_frame_info['pc'].points)
    intermediate_frame_info['lidar_token'] = lidar_rec['token']
    intermediate_frame_info['cs_record'] = nusc.get('calibrated_sensor',
                             lidar_rec['calibrated_sensor_token'])
    sample_token = lidar_rec['sample_token']
    frame = nusc.get('sample', sample_token)
    instance_tokens = [nusc.get('sample_annotation', token)['instance_token'] for token in frame['anns']]
    intermediate_frame_info['pose_record'] = nusc.get('ego_pose', lidar_rec['ego_pose_token'])
    lidar_path, boxes, _ = nusc.get_sample_data(lidar_rec['token'])
    intermediate_frame_info['boxes'] = boxes
    intermediate_frame_info['instance_tokens'] = instance_tokens
    assert len(boxes) == len(instance_tokens) , print('erro')
    return intermediate_frame_info

def intermediate_keyframe_align(nusc: NuScenes, prev_frame_info, ego_frame_info, cur_sample_points):
    ''' align prev_frame points to ego_frame
    return: points (np.array) aligned points of prev_frame
            pc_segs (np.array) label of aligned points of prev_frame
    '''
    prev_frame_info['pc'].points = remove_close(prev_frame_info['pc'].points, (1, 2))
    
    '''
    if  prev_frame_info['lidar_token'] in INTER_STATIC_POINTS:
        static_points = INTER_STATIC_POINTS[prev_frame_info['lidar_token']].copy()
        static_points = prev2ego(static_points, INTER_STATIC_POSE[prev_frame_info['lidar_token']], ego_frame_info)
    else:
        static_points = prev2ego(prev_frame_info['pc'].points, prev_frame_info, ego_frame_info)
    '''
    points = prev2ego(prev_frame_info['pc'].points, prev_frame_info, ego_frame_info)
    pcs = []                                                        
    pcs.append(points)
    return np.concatenate(pcs, axis=-1)

def nonkeykeyframe_align(nusc: NuScenes, prev_frame_info, ego_frame_info, flag='prev', cur_sample_points=None):
    ''' align non keyframe points to ego_frame
    return: points (np.array) aligned points of prev_frame
            pc_segs (np.array) seg of aligned points of prev_frame
    '''
    pcs = []
    start_frame = nusc.get('sample', prev_frame_info['token'])
    end_frame = nusc.get('sample', start_frame[flag])
    # next_frame_info = get_frame_info(end_frame, nusc)
    start_sd_record = nusc.get('sample_data', start_frame['data']['LIDAR_TOP'])
    start_sd_record = nusc.get('sample_data', start_sd_record[flag])
    # end_sd_record = nusc.get('sample_data', end_frame['data']['LIDAR_TOP'])
    # get intermediate frame info
    num = 0
    while start_sd_record['token'] != end_frame['data']['LIDAR_TOP']:
        num +=1
        intermediate_frame_info = get_intermediate_frame_info(nusc, prev_frame_info, start_sd_record, flag)
        pc = intermediate_keyframe_align(nusc, intermediate_frame_info, ego_frame_info, cur_sample_points)
        start_sd_record = nusc.get('sample_data', start_sd_record[flag])
        pcs.append(pc)
    return np.concatenate(pcs, axis=-1)


def prev2ego(points, prev_frame_info, income_frame_info, velocity=None, time_gap=0.0):
    ''' translation prev points to ego frame
    '''
    # prev_sd_rec = nusc.get('sample_data', prev_frame_info['data']['LIDAR_TOP'])

    prev_cs_record = prev_frame_info['cs_record']
    prev_pose_record = prev_frame_info['pose_record']

    points = transform(points, Quaternion(prev_cs_record['rotation']).rotation_matrix, np.array(prev_cs_record['translation']))
    points = transform(points, Quaternion(prev_pose_record['rotation']).rotation_matrix, np.array(prev_pose_record['translation']))

    if velocity is not None:
        points[:3, :] = points[:3, :] + velocity*time_gap

    ego_cs_record = income_frame_info['cs_record']
    ego_pose_record = income_frame_info['pose_record']
    points = transform(points, Quaternion(ego_pose_record['rotation']).rotation_matrix, np.array(ego_pose_record['translation']), inverse=True)
    points = transform(points, Quaternion(ego_cs_record['rotation']).rotation_matrix, np.array(ego_cs_record['translation']), inverse=True)
    return points.copy()


def filter_points_in_ego(points, frame_info, instance_token):
    '''
    filter points in this frame box
    '''
    index = frame_info['instance_tokens'].index(instance_token)
    box = frame_info['boxes'][index]
    # print(f"ego box pos {box.center}")
    box_mask = points_in_box(box, points[:3, :])
    return box_mask

def keyframe_align(prev_frame_info, ego_frame_info):
    ''' align prev_frame points to ego_frame
    return: points (np.array) aligned points of prev_frame
            pc_segs (np.array) seg of aligned points of prev_frame
    '''
    pcs = []

    points = prev_frame_info['pc'].points
    points = prev2ego(points, prev_frame_info, ego_frame_info)
    pcs.append(points.copy())

    return np.concatenate(pcs, axis=-1)

def generate_occupancy_data(sample_token, scene_name, nusc: NuScenes, cur_sample, num_sweeps, save_path='./occupacy/', gt_from: str = 'lidarseg'):
    pcs =[] # for keyframe points
    num_frames = 0
    intermediate_pcs = [] # # for non keyfrme points
    lidar_data = nusc.get('sample_data',
                            cur_sample['data']['LIDAR_TOP'])
    pc = LidarPointCloud.from_file(nusc.dataroot+lidar_data['filename'])
    filename = os.path.split(lidar_data['filename'])[-1]
    lidar_sd_token = cur_sample['data']['LIDAR_TOP']
    
    # align keyframes
    count_prev_frame = 0
    prev_frame = cur_sample.copy()

    while num_sweeps > 0:
        if prev_frame['prev'] == '':
            break
        prev_frame = nusc.get('sample', prev_frame['prev'])
        count_prev_frame += 1
        if count_prev_frame == num_sweeps:
            break
    cur_sample_info = get_frame_info(cur_sample, nusc=nusc)
    # convert prev keyframe to ego frame
    if count_prev_frame > 0:
        prev_info = get_frame_info(prev_frame, nusc)
    pc_points = None
    while count_prev_frame > 0:
        income_info = get_frame_info(frame =prev_frame, nusc=nusc)
        prev_frame = nusc.get('sample', prev_frame['next'])
        prev_info = income_info
        pc_points = keyframe_align(prev_info, cur_sample_info)
        num_frames += 1
        pcs.append(pc_points)
        count_prev_frame -= 1

    # convert next frame to ego frame
    next_frame = cur_sample.copy()
    pc_points = None
    count_next_frame = 0
    while num_sweeps > 0:
        if next_frame['next'] == '':
            break
        next_frame = nusc.get('sample', next_frame['next'])
        count_next_frame += 1
        if count_next_frame == num_sweeps:
            break

    if count_next_frame > 0:
        prev_info = get_frame_info(next_frame, nusc=nusc)

    while count_next_frame > 0:
        
        income_info = get_frame_info(frame=next_frame, nusc=nusc)
        prev_info = income_info
        next_frame =  nusc.get('sample', next_frame['prev'])
        pc_points = keyframe_align(prev_info, cur_sample_info)
        num_frames += 1
        pcs.append(pc_points)
        count_next_frame -= 1
    pcs = np.concatenate(pcs, axis=-1)

    pc.points = np.concatenate((pc.points, pcs), axis=-1)

    range_mask = (pc.points[0,:]<= 60) &  (pc.points[0,:]>=-60)\
     &(pc.points[1,:]<= 60) &  (pc.points[1,:]>=-60)\
      &(pc.points[2,:]<= 10) &  (pc.points[2,:]>=-10)
    pc.points = pc.points[:, range_mask]


    # align nonkeyframe
    count_prev_frame = 0
    prev_frame = cur_sample.copy()

    while num_sweeps > 0:
        if prev_frame['prev'] == '':
            break
        prev_frame = nusc.get('sample', prev_frame['prev'])
        count_prev_frame += 1
        if count_prev_frame == num_sweeps:
            break
    cur_sample_info = get_frame_info(cur_sample, nusc=nusc)
    # convert prev frame to ego frame
    if count_prev_frame > 0:
        prev_info = get_frame_info(prev_frame, nusc)
    while count_prev_frame > 0:
        income_info = get_frame_info(frame =prev_frame, nusc=nusc)
        prev_frame = nusc.get('sample', prev_frame['next'])
        prev_info = income_info
        intermediate_pc = nonkeykeyframe_align(nusc, prev_info, cur_sample_info, 'next', pc.points)
        num_frames += 1
        intermediate_pcs.append(intermediate_pc)
        count_prev_frame -= 1

    next_frame = cur_sample.copy()
    count_next_frame = 0
    while num_sweeps > 0:
        if next_frame['next'] == '':
            break
        next_frame = nusc.get('sample', next_frame['next'])
        count_next_frame += 1
        if count_next_frame == num_sweeps:
            break

    if count_next_frame > 0:
        prev_info = get_frame_info(next_frame, nusc=nusc)

    while count_next_frame > 0:
        
        income_info = get_frame_info(frame =next_frame, nusc=nusc)
        prev_info = income_info
        next_frame =  nusc.get('sample', next_frame['prev'])
        intermediate_pc = nonkeykeyframe_align(nusc, prev_info, cur_sample_info, 'prev', pc.points)
        num_frames += 1
        intermediate_pcs.append(intermediate_pc)
        count_next_frame -= 1
    intermediate_pcs = np.concatenate(intermediate_pcs, axis=-1)
    pc.points = np.concatenate((pc.points, intermediate_pcs), axis=1)
    # removed too dense point
    raw_point = pc.points.transpose(1,0)[:,:3]

    assert pc.points.transpose(1,0)[:,3:].max()<=255
    pcd=o3d.open3d.geometry.PointCloud()
    
    pcd.points= o3d.open3d.utility.Vector3dVector(raw_point)
    pcd_new = o3d.geometry.PointCloud.voxel_down_sample(pcd, 0.2)
    new_points = np.asarray(pcd_new.points)

    range_mask = (new_points[:,0]<= 60) &  (new_points[:,0]>=-60)\
     &(new_points[:,1]<= 60) &  (new_points[:,1]>=-60)\
      &(new_points[:,2]<= 10) &  (new_points[:,2]>=-10)
    save_points = new_points[range_mask]
    new_points = new_points[range_mask]
    new_points = new_points.astype(np.float16)

    # lidar coords to ego coords
    trans = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    save_points = (trans @ save_points.T).T
    
    # save occupancy
    max_bound = np.asarray([51.2, 51.2, 3])  # 51.2 51.2 3
    min_bound = np.asarray([-51.2, -51.2, -5])  # -51.2 -51.2 -5
    # get grid index
    crop_range = max_bound - min_bound
    cur_grid_size = np.asarray([200, 200, 16])                 # 200, 200, 16
    intervals = crop_range / (cur_grid_size - 1)

    if (intervals == 0).any(): 
        print("Zero interval!")
    grid_ind = (np.floor((np.clip(save_points, min_bound, max_bound) - min_bound) / intervals)).astype(np.int) 

    # process labels
    processed_label = np.zeros([200, 200, 16], dtype=np.uint8)
    label_voxel_pair = grid_ind[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
    for i in range(label_voxel_pair.shape[0]):
        cur_ind = label_voxel_pair[i, :3]
        processed_label[cur_ind[0], cur_ind[1], cur_ind[2]] = 1

    mask_lidar_fake = np.zeros([200, 200, 16], dtype=np.uint8)
    mask_camera_fake = np.zeros([200, 200, 16], dtype=np.uint8)
    save_dir = os.path.join(save_path, 'gts', scene_name, sample_token)
    save_path_ = os.path.join(save_dir, 'labels_binary_sweep2.npz')
    np.savez_compressed(save_path_, semantics=processed_label, mask_lidar=mask_lidar_fake, mask_camera=mask_camera_fake)

    return pc.points

def convert2occupy(dataroot,
                        save_path, num_sweeps=10,):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cnt = 0
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    for scene in nusc.scene:
        INTER_STATIC_POINTS.clear()
        INTER_STATIC_LABEL.clear()
        INTER_STATIC_POSE.clear()
        sample_token = scene['first_sample_token']
        cur_sample = nusc.get('sample', sample_token)
        scene_name = scene['name']
        print('scene_name',scene_name)
        #import pdb; pdb.set_trace()
        while True:
            cnt += 1
            print(cnt)
            print('scene_name',scene_name)
            print('cur_sample',cur_sample["token"])
            cur_token = cur_sample["token"]
            save_dir = os.path.join(save_path, 'gts', scene_name, cur_token)
            os.makedirs(save_dir, exist_ok=True)
            generate_occupancy_data(cur_token, scene_name, nusc, cur_sample, num_sweeps, save_path=save_path)
            if cur_sample['next'] == '':
                break
            cur_sample = nusc.get('sample', cur_sample['next'])

if __name__ == "__main__":
    args = parse_args()
    convert2occupy(args.dataroot, args.save_path, args.num_sweeps)

