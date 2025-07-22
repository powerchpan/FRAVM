import joblib
import numpy as np
import json

#save_path = '/home/user/ply/data/monocular/sts/xupt_students/front/1/metadata.json'
def get_camera_parameters(pred_cam, bbox):
    FOCAL_LENGTH = 5000.
    CROP_SIZE = 224

    bbox_cx, bbox_cy, bbox_w, bbox_h = bbox
    assert bbox_w == bbox_h

    bbox_size = bbox_w
    bbox_x = bbox_cx - bbox_w / 2.
    bbox_y = bbox_cy - bbox_h / 2.

    scale = bbox_size / CROP_SIZE

    cam_intrinsics = np.eye(3)
    cam_intrinsics[0, 0] = FOCAL_LENGTH * scale
    cam_intrinsics[1, 1] = FOCAL_LENGTH * scale
    cam_intrinsics[0, 2] = bbox_size / 2. + bbox_x 
    cam_intrinsics[1, 2] = bbox_size / 2. + bbox_y

    cam_s, cam_tx, cam_ty = pred_cam
    trans = [cam_tx, cam_ty, 2*FOCAL_LENGTH/(CROP_SIZE*cam_s + 1e-9)]

    cam_extrinsics = np.eye(4)
    cam_extrinsics[:3, 3] = trans

    return cam_intrinsics, cam_extrinsics
    
    

output = joblib.load('/home/user/ply/data/five_sts/keypoints_3d/hospital_20230410/side_good/cs-side-1/vibe_output.pkl')
print(len(output))
poses = output[1]['pose']
betas = output[1]['betas']
pred_cams = output[1]['pred_cam']
bboxes = output[1]['bboxes']
frame_ids = output[1]['frame_ids']
data_dic = {}
i = 0
while i< len(frame_ids):
    frame_id = frame_ids[i]
    dic_name = str(frame_id)
    pose = poses[i]
    beta = betas[i]
    pred_cam = pred_cams[i]
    bbox = bboxes[i]
    cam_intrinsics, cam_extrinsics = get_camera_parameters(pred_cam, bbox)
    dic = {}
    pose = pose.tolist()
    beta = beta.tolist()
    cam_intrinsics = cam_intrinsics.tolist()
    cam_extrinsics = cam_extrinsics.tolist()
    dic['poses'] = pose
    dic['betas'] = beta
    dic['cam_intrinsics'] = cam_intrinsics
    dic['cam_extrinsics'] = cam_extrinsics
    data_dic[dic_name] = dic
    i+=1
with open(save_path, 'w') as f:
    f.write(json.dumps(data_dic, ensure_ascii=False))

#for k,v in output[1].items():
#    print(k)
#    frame_data = {}
#    if(k=='poses'):
#        frame_data['poses'] = v 
#    elif(k=='betas'):
#        frame_data['betas'] = v
#    elif(k=='pred_cam'):
#         bbox = output[1]['bboxes']
#         print(bbox.shape)
#         cam_intrinsics, cam_extrinsics = get_camera_parameters(k, bbox)
#         frame_data['cam_intrinsics'] = cam_intrinsics
#         frame_data['cam_extrinsics'] = cam_extrinsics
    
    #print(k,v.shape)
    
    




