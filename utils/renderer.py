import os

os.environ["DISPLAY"] = ":0"

import numpy as np
import pyrender
import trimesh


def get_colors():
    colors = {
        'pink': np.array([197, 27, 125]),
        'light_pink': np.array([233, 163, 201]),
        'light_green': np.array([161, 215, 106]),
        'green': np.array([77, 146, 33]),
        'red': np.array([215, 48, 39]),
        'light_red': np.array([252, 146, 114]),
        'light_orange': np.array([252, 141, 89]),
        'purple': np.array([118, 42, 131]),
        'light_purple': np.array([175, 141, 195]),
        'light_blue': np.array([145, 191, 219]),
        'blue': np.array([69, 117, 180]),
        'gray': np.array([130, 130, 130]),
        'white': np.array([255, 255, 255]),
        'pinkish': np.array([204, 77, 77]),
    }
    return colors


class AMASSRenderer:
    def __init__(self, focal_length=5000.0, w=512, h=512, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h, point_size=1.0)

        self.focal_length = focal_length
        self.camera_center = [w // 2, h // 2]
        self.faces = faces
        self.pred_mesh_color = get_colors()['gray']
        self.target_mesh_color = get_colors()['red']
        self.cam_r = np.eye(3)

    def overlay_image(self, img, pred_verts, cam_t, sideview=False, rot_angle=None):
        pred_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(
                self.pred_mesh_color[0] / 255., self.pred_mesh_color[1] / 255., self.pred_mesh_color[2] / 255., 1.0))

        pred_mesh = trimesh.Trimesh(pred_verts, self.faces)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        pred_mesh.apply_transform(rot)

        if sideview:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(270), [0, 1, 0])
            pred_mesh.apply_transform(rot)

        if rot_angle is not None:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0])
            pred_mesh.apply_transform(rot)

        pyrender_pred_mesh = pyrender.Mesh.from_trimesh(pred_mesh, material=pred_material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))

        scene.add(pyrender_pred_mesh, 'pred_mesh')

        cam_t[0] *= -1.
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = self.cam_r
        cam_pose[:3, 3] = cam_t

        cam = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length, cx=self.camera_center[0],
                                        cy=self.camera_center[1])

        scene.add(cam, pose=cam_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32)
        valid_mask = (rend_depth > 0)[:, :, None]

        output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * img

        return output_img.astype(np.uint8)
