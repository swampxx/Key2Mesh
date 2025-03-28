import glob
import os
import pickle

import cv2
import joblib
import numpy as np

from utils.transforms import swap_axes


def preprocess_3dpw(dataset_path, split, out_file):
    # data we'll save
    output = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'shape': [],
        'pose': [],
        'img_name': [],
    }

    split_dir = os.path.join(dataset_path, "sequenceFiles", split)
    print(split_dir)
    for filename in sorted(glob.glob(f"{split_dir}/*.pkl")):
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        num_people = len(data["poses"])
        num_frames = len(data["poses"][0])

        genders = data['genders']
        smpl_shape = data["betas"]

        seq_name = str(data['sequence'])
        imgnames = np.array([
            f"imageFiles/{seq_name}/image_{i:05}.jpg"
            for i in range(num_frames)
        ])
        for i in range(num_people):
            valid_campose = data["campose_valid"][i].astype(bool)

            # Same as SPIN, we consider valid frames only
            valid_imgnames = imgnames[valid_campose]
            valid_smpl_pose = data["poses"][i][valid_campose]
            valid_smpl_shape = np.tile(smpl_shape[i][:10],
                                       (len(valid_smpl_pose), 1))
            valid_genders = np.tile(genders[i], len(valid_smpl_pose))

            valid_poses2d = data["poses2d"][i][valid_campose].transpose(0, 2, 1)
            valid_poses3d = data["jointPositions"][i][valid_campose].reshape(-1, 24, 3)

            # transform global poses
            extrinsics = data["cam_poses"][valid_campose][:, :3, :3]
            for j in range(len(extrinsics)):
                global_rot_mat = cv2.Rodrigues(valid_smpl_pose[j, :3])[0]
                valid_smpl_pose[j, :3] = cv2.Rodrigues(np.dot(extrinsics[j], global_rot_mat))[0].T[0]

            valid_smpl_pose = swap_axes(valid_smpl_pose, np.pi, 0, 0)

            valid_seq_len = valid_poses2d.shape[0]

            output['vid_name'].append(np.array([f'{seq_name}_{i}'] * valid_seq_len))
            output['frame_id'].append(np.arange(0, num_frames)[valid_campose])
            output['img_name'].append(np.array(imgnames[valid_campose]))
            output['joints3D'].append(np.array(valid_poses3d))
            output['joints2D'].append(np.array(valid_poses2d))
            output['shape'].append(np.array(valid_smpl_shape))
            output['pose'].append(np.array(valid_smpl_pose))

    for k, v in output.items():
        output[k] = np.concatenate(v)
        print(f'{k}, {output[k].shape}')

    joblib.dump(output, out_file)


if __name__ == "__main__":
    preprocess_3dpw("./data/3DPW/", "test", "./data/3DPW_test.pt")
