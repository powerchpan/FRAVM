# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.utils.smooth_pose import smooth_pose
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker

from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)
from dtw_similiarity import choose_keypoints
from model_msgcn import Trainer
from get_feature import get_feature
import torch.nn.functional as F

from get_individual_height import get_individual_height
from get_feature_2 import get_all_time
from get_feature_2 import get_sts_times
from get_feature_2 import get_each_sitting
from get_feature_2 import get_each_sitstand
from get_feature_2 import get_each_standing
from get_feature_2 import get_each_standsit
from get_feature_2 import get_sitstand_velocity
from get_feature_2 import get_standsit_velocity
from get_feature_2 import get_sts_velocity
from get_feature_2 import get_sts_velocity
from get_feature_2 import get_mean_sitstand_velocity
from get_feature_2 import get_mean_standsit_velocity
from get_feature_2 import get_mean_sts_velocity
from get_feature_2 import get_max_sitstand_velocity
from get_feature_2 import get_max_standsit_velocity
from get_feature_2 import get_max_sts_velocity
from get_feature_2 import get_min_sitstand_velocity
from get_feature_2 import get_min_standsit_velocity
from get_feature_2 import get_min_sts_velocity
from get_feature_2 import get_sitstand_accelaration
from get_feature_2 import get_standsit_accelaration
from get_feature_2 import get_each_sts_accelaration
from get_feature_2 import get_sts_accelaration
from get_feature_2 import get_mean_sitting
from get_feature_2 import get_mean_standing
from get_feature_2 import get_single_sts_duration
from get_feature_2 import get_min_sts_velocity




standard_dir = '/home/user/ply/code/VIBE-master/standard_data.npy'
target_path = '/home/user/ply/code/VIBE-master/data_msgcn/sts/keypoints25_3d/1.npy'

MIN_NUM_FRAMES = 25


def get_parameters_sts(keypoints, results):
    print(results)
    last_frame_list,sts_times = get_sts_times(results)
    each_sitting_time = get_each_sitting(results)
    each_sitstand_time = get_each_sitstand(results)
    each_standsit_time = get_each_standsit(results)
    sitstand_velocity = get_sitstand_velocity(keypoints,results)
    standsit_velocity = get_standsit_velocity(keypoints,results)
    sts_velocity = get_sts_velocity(keypoints,results)
    mean_sitstand_velocity = get_mean_sitstand_velocity(keypoints,results)
    mean_standsit_velocity = get_mean_standsit_velocity(keypoints,results)
    mean_sts_velocity = get_mean_sts_velocity(keypoints,results)
    max_sitstand_velocity = get_max_sitstand_velocity(keypoints,results)
    max_standsit_velocity = get_max_standsit_velocity(keypoints,results)
    max_sts_velocity = get_max_sts_velocity(keypoints,results)
    min_sitstand_velocity = get_min_sitstand_velocity(keypoints,results)
    min_standsit_velocity = get_min_standsit_velocity(keypoints,results)   
    min_sts_velocity = get_min_sts_velocity(keypoints,results)   
    sitstand_accelaration = get_sitstand_accelaration(keypoints,results)
    standsit_accelaration = get_standsit_accelaration(keypoints,results)
    each_sts_accelaration = get_each_sts_accelaration(keypoints,results)
    sts_accelaration = get_sts_accelaration(keypoints,results)    
    mean_sitting_time = get_mean_sitting(results)
    mean_standing_time = get_mean_standing(results)      
    single_sts_duration = get_single_sts_duration(results)
    
    print("sts_times:"+str(sts_times))
    print("last_frame_list:"+str(last_frame_list))
    print("each_sitting_time:"+str(each_sitting_time))
    print("each_sitstand_time:"+str(each_sitstand_time))
    print("each_standsit_time:"+str(each_standsit_time))
    print("sitstand_velocity:"+str(sitstand_velocity))
    print("standsit_velocity:"+str(standsit_velocity))
    print("sts_velocity:"+str(sts_velocity))
    print("mean_sitstand_velocity:"+str(mean_sitstand_velocity))
    print("mean_sts_velocity:"+str(mean_sts_velocity))
    print("max_sitstand_velocity:"+str(max_sitstand_velocity))
    print("max_standsit_velocity:"+str(max_standsit_velocity))
    print("max_sts_velocity:"+str(max_sts_velocity))
    print("min_sitstand_velocity:"+str(min_sitstand_velocity))
    print("min_standsit_velocity:"+str(min_standsit_velocity))
    print("min_sts_velocity:"+str(min_sts_velocity))
    print("sitstand_accelaration:"+str(sitstand_accelaration))
    print("standsit_accelaration:"+str(standsit_accelaration))
    print("each_sts_accelaration:"+str(each_sts_accelaration))
    print("sts_accelaration:"+str(sts_accelaration))
    print("mean_sitting_time:"+str(mean_sitting_time))
    print("mean_standing_time:"+str(mean_standing_time))
    print("single_sts_duration:"+str(single_sts_duration))
    
    
   # each_sitting_time_arr = np.array(each_sitting_time)
   # each_sitstand_time_arr = np.array(each_sitstand_time)
   # eeach_standsit_time_arr = np.array(each_standsit_time)
   # sitstand_velocity_arr = np.array(sitstand_velocity)
   # standsit_velocity_arr = np.array(standsit_velocity)
   # sts_velocity_arr = np.array(sts_velocity)
   # mean_sitstand_velocity_arr = np.array(mean_sitstand_velocity)
   # mean_standsit_velocity_arr = np.array(mean_standsit_velocity)
   # mean_sts_velocity_arr = np.array(mean_sts_velocity)
   # max_sitstand_velocity_arr = np.array(max_sitstand_velocity)
   # max_standsit_velocity_arr = np.array(max_standsit_velocity)
   # max_sts_velocity_arr = np.array(max_sts_velocity)
   # min_sitstand_velocity_arr = np.array(min_sitstand_velocity)
   # min_standsit_velocity_arr = np.array(min_standsit_velocity)
   # min_sts_velocity_arr = np.array(min_sts_velocity)
   # sitstand_accelaration_arr = np.array(sitstand_accelaration)
   # standsit_accelaration_arr = np.array(standsit_accelaration)
   # each_sts_accelaration_arr = np.array(each_sts_accelaration)
   # mean_sitting_time_arr = np.array(mean_sitting_time)
   # mean_standing_time_arr = np.array(mean_standing_time)
   # single_sts_duration_arr = np.array(single_sts_duration)
    
    #print(mean_sitstand_velocity_arr)
    parameters_list = []
    parameters_list.append(each_sitting_time)
    parameters_list.append(each_sitstand_time)
    parameters_list.append(each_standsit_time)
    parameters_list.append(sitstand_velocity)
    parameters_list.append(standsit_velocity)
    parameters_list.append(sts_velocity)
    parameters_list.append(mean_sitstand_velocity)
    parameters_list.append(mean_standsit_velocity)
    parameters_list.append(mean_sts_velocity)
    parameters_list.append(max_sitstand_velocity)
    parameters_list.append(max_standsit_velocity)
    parameters_list.append(max_sts_velocity)
    parameters_list.append(min_sitstand_velocity)
    parameters_list.append(min_standsit_velocity)
    parameters_list.append(min_sts_velocity)
    parameters_list.append(sitstand_accelaration)
    parameters_list.append(standsit_accelaration)
    parameters_list.append(each_sts_accelaration)
    parameters_list.append(mean_sitting_time)
    parameters_list.append(mean_standing_time)
    parameters_list.append(single_sts_duration)
    
    parameters_all_list = []
    i = 0
    while(i<len(parameters_list)):
        parameters_index = parameters_list[i]
        if isinstance(parameters_index, float):
            parameters_all_list.append(parameters_index)
        else:
            j = 0
            while(j<len(parameters_index)):
                #print(parameters_index[j])
                parameters_all_list.append(parameters_index[j])
                j+=1
        i+=1
    #print(parameters_all_list)
    parameters_array = np.array(parameters_all_list) #parameters_array=each_sitting_time_arr+each_sitstand_time_arr+eeach_standsit_time_arr+sitstand_velocity_arr+standsit_velocity_arr+sts_velocity_arr+mean_sitstand_velocity_arr+mean_standsit_velocity_arr+mean_sts_velocity_arr+max_sitstand_velocity_arr+max_standsit_velocity_arr+max_sts_velocity_arr+min_sitstand_velocity_arr+min_standsit_velocity_arr+min_sts_velocity_arr++standsit_accelaration_arr+each_sts_accelaration_arr+mean_sitting_time_arr+mean_standing_time_arr+single_sts_duration_arr
   # parameters_array = np.concatenate((each_sitting_time_arr, each_sitstand_time_arr, eeach_standsit_time_arr, sitstand_velocity_arr, standsit_velocity_arr, sts_velocity_arr, mean_sitstand_velocity_arr, mean_standsit_velocity_arr, mean_sts_velocity_arr, max_sitstand_velocity_arr, max_standsit_velocity_arr, max_sts_velocity_arr, min_sitstand_velocity_arr, min_standsit_velocity_arr, min_sts_velocity_arr, sitstand_accelaration_arr, standsit_accelaration_arr, each_sts_accelaration_arr, mean_sitting_time_arr, mean_standing_time_arr, single_sts_duration_arr), axis=0)
    #print(parameters_array.shape)    
    return parameters_array
    
def correct_results_segmentation(results):
    i=0
    while((i<len(results)-1)&(i>0)):
        if((results[i-1]==results[i+1])&(results[i]!=results[i+1])):
            results[i] = results[i+1]
        i+=1
    return results


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    video_file = args.vid_file

    # ========= [Optional] download the youtube video ========= #
    if video_file.startswith('https://www.youtube.com'):
        print(f'Donwloading YouTube video \"{video_file}\"')
        video_file = download_youtube_clip(video_file, '/tmp')

        if video_file is None:
            exit('Youtube url is not valid!')

        print(f'YouTube Video has been downloaded to {video_file}...')

    if not os.path.isfile(video_file):
        exit(f'Input video \"{video_file}\" does not exist!')

    output_path = os.path.join(args.output_folder, os.path.basename(video_file).replace('.mp4', ''))
    os.makedirs(output_path, exist_ok=True)

    image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)

    print(f'Input video number of frames {num_frames}')
    orig_height, orig_width = img_shape[:2]

    total_time = time.time()

    # ========= Run tracking ========= #
    bbox_scale = 1.1
    if args.tracking_method == 'pose':
        if not os.path.isabs(video_file):
            video_file = os.path.join(os.getcwd(), video_file)
        tracking_results = run_posetracker(video_file, staf_folder=args.staf_dir, display=args.display)
    else:
        # run multi object tracker
        mot = MPT(
            device=device,
            batch_size=args.tracker_batch_size,
            display=args.display,
            detector_type=args.detector,
            output_format='dict',
            yolo_img_size=args.yolo_img_size,
        )
        tracking_results = mot(image_folder)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    # ========= Define VIBE model ========= #
    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    # ========= Load pretrained weights ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    # ========= Run VIBE on each person ========= #
    print(f'Running VIBE on each tracklet...')
    vibe_time = time.time()
    vibe_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None

        if args.tracking_method == 'bbox':
            bboxes = tracking_results[person_id]['bbox']
        elif args.tracking_method == 'pose':
            joints2d = tracking_results[person_id]['joints2d']

        frames = tracking_results[person_id]['frames']

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        dataloader = DataLoader(dataset, batch_size=args.vibe_batch_size, num_workers=16)

        with torch.no_grad():

            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d, norm_joints2d = [], [], [], [], [], [], []

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.unsqueeze(0)
                batch = batch.to(device)

                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]

                pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
                pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
                smpl_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2))


            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            smpl_joints2d = torch.cat(smpl_joints2d, dim=0)
            del batch

        # ========= [Optional] run Temporal SMPLify to refine the results ========= #
        if args.run_smplify and args.tracking_method == 'pose':
            norm_joints2d = np.concatenate(norm_joints2d, axis=0)
            norm_joints2d = convert_kps(norm_joints2d, src='staf', dst='spin')
            norm_joints2d = torch.from_numpy(norm_joints2d).float().to(device)

            # Run Temporal SMPLify
            update, new_opt_vertices, new_opt_cam, new_opt_pose, new_opt_betas, \
            new_opt_joints3d, new_opt_joint_loss, opt_joint_loss = smplify_runner(
                pred_rotmat=pred_pose,
                pred_betas=pred_betas,
                pred_cam=pred_cam,
                j2d=norm_joints2d,
                device=device,
                batch_size=norm_joints2d.shape[0],
                pose2aa=False,
            )

            # update the parameters after refinement
            print(f'Update ratio after Temporal SMPLify: {update.sum()} / {norm_joints2d.shape[0]}')
            pred_verts = pred_verts.cpu()
            pred_cam = pred_cam.cpu()
            pred_pose = pred_pose.cpu()
            pred_betas = pred_betas.cpu()
            pred_joints3d = pred_joints3d.cpu()
            pred_verts[update] = new_opt_vertices[update]
            pred_cam[update] = new_opt_cam[update]
            pred_pose[update] = new_opt_pose[update]
            pred_betas[update] = new_opt_betas[update]
            pred_joints3d[update] = new_opt_joints3d[update]

        elif args.run_smplify and args.tracking_method == 'bbox':
            print('[WARNING] You need to enable pose tracking to run Temporal SMPLify algorithm!')
            print('[WARNING] Continuing without running Temporal SMPLify!..')

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()
        smpl_joints2d = smpl_joints2d.cpu().numpy()

        # Runs 1 Euro Filter to smooth out the results
        if args.smooth:
            min_cutoff = args.smooth_min_cutoff # 0.004
            beta = args.smooth_beta # 1.5
            print(f'Running smoothing on person {person_id}, min_cutoff: {min_cutoff}, beta: {beta}')
            pred_verts, pred_pose, pred_joints3d = smooth_pose(pred_pose, pred_betas,
                                                               min_cutoff=min_cutoff, beta=beta)

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        joints2d_img_coord = convert_crop_coords_to_orig_img(
            bbox=bboxes,
            keypoints=smpl_joints2d,
            crop_size=224,
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'joints2d_img_coord': joints2d_img_coord,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        vibe_results[person_id] = output_dict

    del model

    end = time.time()
    fps = num_frames / (end - vibe_time)

    print(f'VIBE FPS: {fps:.2f}')
    total_time = time.time() - total_time
    print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
    print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

    print(f'Saving output results to \"{os.path.join(output_path, "vibe_output.pkl")}\".')

    joblib.dump(vibe_results, os.path.join(output_path, "vibe_output.pkl"))

    if not args.no_render:
        # ========= Render results as a single video ========= #
        renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=args.wireframe)

        output_img_folder = f'{image_folder}_output'
        os.makedirs(output_img_folder, exist_ok=True)

        print(f'Rendering output video, writing frames to {output_img_folder}')

        # prepare results for rendering
        frame_results = prepare_rendering_results(vibe_results, num_frames)
        mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

        image_file_names = sorted([
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)

            if args.sideview:
                side_img = np.zeros_like(img)

            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data['verts']
                frame_cam = person_data['cam']

                mc = mesh_color[person_id]

                mesh_filename = None

                if args.save_obj:
                    mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
                    os.makedirs(mesh_folder, exist_ok=True)
                    mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

                img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    mesh_filename=mesh_filename,
                )

                if args.sideview:
                    side_img = renderer.render(
                        side_img,
                        frame_verts,
                        cam=frame_cam,
                        color=mc,
                        angle=270,
                        axis=[0,1,0],
                    )

            if args.sideview:
                img = np.concatenate([img, side_img], axis=1)

            cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

            if args.display:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if args.display:
            cv2.destroyAllWindows()

        # ========= Save rendered video ========= #
        vid_name = os.path.basename(video_file)
        save_name = f'{vid_name.replace(".mp4", "")}_vibe_result.mp4'
        save_name = os.path.join(output_path, save_name)
        print(f'Saving result video to {save_name}')
        images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
        shutil.rmtree(output_img_folder)

    shutil.rmtree(image_folder)
    print('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str,
                        help='input video path or youtube link')

    parser.add_argument('--output_folder', type=str,
                        help='output folder to write results')

    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--staf_dir', type=str, default='/home/mkocabas/developments/openposetrack',
                        help='path to directory STAF pose tracking method installed.')

    parser.add_argument('--vibe_batch_size', type=int, default=450,
                        help='batch size of VIBE')

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--run_smplify', action='store_true',
                        help='run smplify for refining the results, you need pose tracking to enable it')

    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    parser.add_argument('--smooth', action='store_true',
                        help='smooth the results to prevent jitter')

    parser.add_argument('--smooth_min_cutoff', type=float, default=0.004,
                        help='one euro filter min cutoff. '
                             'Decreasing the minimum cutoff frequency decreases slow speed jitter')

    parser.add_argument('--smooth_beta', type=float, default=0.7,
                        help='one euro filter beta. '
                             'Increasing the speed coefficient(beta) decreases speed lag.')
    parser.add_argument('--action', default='predict')
    parser.add_argument('--dataset', default="sts")

    args = parser.parse_args()

    main(args)
    print('Target  matching---------------------')
    video_file = args.vid_file
    output_path = os.path.join(args.output_folder, os.path.basename(video_file).replace('.mp4', ''))
    raw_data_path = os.path.join(output_path, "vibe_output.pkl")
    output = joblib.load(raw_data_path)
    all_keypoints = []
    i=0
    for j in output.keys():
        keypoints_3d = output[j]['joints3d']
        keypoints_3d_25 = keypoints_3d[:,0:25,:] 
        all_keypoints.append(keypoints_3d_25)
        i+=1
    np.save(target_path, all_keypoints)
   # standard_keypoints = joblib.load(standard_dir)
   # standard_keypoints = standard_keypoints[1]['joints3d']
   # standard_keypoints = standard_keypoints[:,0:25,:] 
   # min_index, keypoints, keypoints_max = choose_keypoints(standard_dir, all_keypoints)
    min_index, keypoints, keypoints_max = choose_keypoints(standard_dir, target_path)
    keypoints = np.array(keypoints)
    #print(keypoints)
    keypoints = keypoints.squeeze()
    keypoints_arr = np.zeros((len(keypoints), 25, 3))
    i = 0
    for element in keypoints:
        keypoints_arr[i] = element
        i+=1
    np.save('/home/user/ply/code/VIBE-master/data_msgcn/sts/features7/1.npy', keypoints_arr)
    feature = get_feature('/home/user/ply/code/VIBE-master/data_msgcn/sts/features7/1.npy')
    feature = F.normalize(feature)
    np.save('/home/user/ply/code/VIBE-master/data_msgcn/sts/features7/1_input.npy', feature)
    print('Predicting labels--------------------')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    i = 1
    model_dir = "./models_msgcn/"+args.dataset+"/split_"+str(i)
    results_dir = "./results_msgcn/"+args.dataset+"/split_"+str(i)
    num_epochs = 100
    features_path = "./data_msgcn/" + args.dataset + "/features7/"
    vid_list_file_tst = "./data_msgcn/" + args.dataset + "/splits_loso_validation/test.split" + str(i) + ".bundle"
    mapping_file = "./data_msgcn/sts/" +"/mapping.txt"
    actions_dict = dict()
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    num_classes = len(actions_dict)
    sample_rate = 1
    dil = [1,2,4,8,16,32,64,128,256,512]
    num_layers_RF = 10
    num_stages = 4
    num_f_maps = 64
    features_dim = 6
    trainer = Trainer(dil, num_layers_RF, num_stages, num_f_maps, features_dim, num_classes)
    if args.action == "predict":
        trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)
    res_path = results_dir+'/1'
    results = np.loadtxt(res_path)
    results = np.array(results)
    results = correct_results_segmentation(results)
    #print(results)
    
    keypoints = np.load('/home/user/ply/code/VIBE-master/data_msgcn/sts/features7/1.npy')
    all_time = get_all_time(results)
    parameters = get_parameters_sts(keypoints, results)
    #print(keypoints_3d)
    
    print(len(results))
    print(parameters)
    print(len(parameters))
    

    
