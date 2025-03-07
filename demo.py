import json
from collections import defaultdict
from pathlib import Path

import cv2
import hydra
import numpy as np
import pytorch_lightning as pl
import torch

from models.key2mesh import Key2Mesh
from utils.geometry import estimate_translation_np
from utils.pose_util import normalize_p2d
from utils.renderer import AMASSRenderer
from utils.transforms import euler_to_rot


def get_bbox(kp2d, conf):
    scaleFactor = 1.6

    part = kp2d[:, :]
    part = part[conf > 0, :]
    bbox = [min(part[:, 0]), min(part[:, 1]),
            max(part[:, 0]), max(part[:, 1])]
    center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
    scale = scaleFactor * max(bbox[2] - bbox[0], bbox[3] - bbox[1])

    return center, scale


def read_openpose_results(op_outs):
    vid_kps = defaultdict(list)
    for kp_json in sorted(op_outs.glob("*keypoints.json"), key=lambda x: int(x.name.split("_keypoints")[0])):
        with open(kp_json, "r") as f:
            kps = json.load(f)
        keypoints_all = [np.array(i["pose_keypoints_2d"]).reshape(18, 3) for i in kps["people"]]
        biggest_box_idx = np.argmax([get_bbox(person[:, :2], person[:, 2])[1] for person in keypoints_all])

        kps = keypoints_all[biggest_box_idx]
        bbox = get_bbox(kps[:, :2], kps[:, 2])

        vid_kps["kp"].append(kps)
        vid_kps["bbox_center"].append(bbox[0])
        vid_kps["bbox_scale"].append(bbox[1])

    return {k: np.array(v) for k, v in vid_kps.items()}


@hydra.main(config_path="configs", config_name="demo")
def main(opt):
    pl.seed_everything(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    checkpoint = f"experiments/{opt.exp}/{opt.run}/ckpts/best-model.ckpt"
    model = Key2Mesh.load_from_checkpoint(checkpoint, strict=False, opt=opt.model,
                                          loss_opt=opt.loss,
                                          train_opt=opt.train)
    model.eval()

    input_root = Path(opt.input.dir)
    raw_imgs = input_root / "imgs"
    openpose_outs = input_root / "openpose_outs"
    outs = input_root / "outs"
    img_size = opt.input.img_size

    renderer = AMASSRenderer(w=img_size[0], h=img_size[1], faces=model.body_model.faces)

    openpose_kp2d = read_openpose_results(openpose_outs)

    kps = torch.from_numpy(openpose_kp2d["kp"][:, :, :2]).float()
    conf = torch.from_numpy(openpose_kp2d["kp"][:, :, 2]).float()

    inp = normalize_p2d(kps, confidence=conf, threshold=0.1)
    smpl_estimations = model(inp)

    with torch.no_grad():
        joint3d_pr, smpl_out_pr = model.get_joint3d(smpl_estimations["poses"], smpl_estimations["betas"],
                                                    protocol='COCO18', ret_smpl_out=True)

    for img_idx, img_p in enumerate(sorted(raw_imgs.glob("*.jpg"), key=lambda x: int(x.name.split(".jpg")[0]))):
        img = cv2.imread(str(img_p))
        j2d = kps[img_idx].cpu().numpy()

        # Swap axis
        mat = euler_to_rot([np.pi], [0], [0])[0]
        j3d_pr = np.dot(joint3d_pr[img_idx].cpu().numpy(), mat.T)
        verts = np.dot(smpl_out_pr.vertices[img_idx].cpu().numpy(), mat.T)

        # Estimate camera translation corresponding to detected 2D keypoints and predicted 3D keypoints.
        cam_t = estimate_translation_np(j3d_pr, j2d, conf[img_idx], focal_length=5000.0, img_size=IMG_SIZE)
        img_overlay = renderer.overlay_image(img, verts, cam_t)
        img_p = outs / f"{img_p.stem}.jpg"
        cv2.imwrite(str(img_p), img_overlay)


if __name__ == '__main__':
    main()
