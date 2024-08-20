from itertools import chain
from typing import Dict, Any

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from smplx import SMPL

from utils import geometry
from utils.camera import PinholeCamera
from utils.network import MLPFeatureExtractor, SMPLHead, DomainCritic, \
    set_requires_grad
from utils.pose_util import compute_MPJPE, normalize_p2d
from utils.smpl import SMPLJointMapper


class Key2Mesh(pl.LightningModule):
    def __init__(self, opt=None, loss_opt=None, train_opt=None):
        super().__init__()
        self.automatic_optimization = False

        self.opt = opt
        self.loss_opt = loss_opt
        self.train_opt = train_opt

        self.mode = self.train_opt.mode
        self.occ_prob = train_opt.occ_prob

        self.feature_extractor_s, self.feature_extractor_t, self.smpl_head, self.domain_head = self.get_model()
        # select main feature extractor based on training mode
        self.feature_extractor = self.feature_extractor_s if self.mode == 'source' else self.feature_extractor_t

        if self.mode == "target":
            # network
            self.training_modules = [
                'feature_extractor',
            ]
            self.feature_extractor_s.eval()
            set_requires_grad(self.feature_extractor_s, requires_grad=False)

            self.smpl_head.eval()
            set_requires_grad(self.smpl_head, requires_grad=False)

        else:
            self.training_modules = [
                'feature_extractor',
                'smpl_head'
            ]

        self.smpl_heads = {
            'fs': self.smpl_head,
        }

        self.camera = PinholeCamera(
            fx=256, fy=256, cx=0, cy=0, R=np.eye(3), t=(0, 0, -6))

        self.body_model = SMPL(model_path=hydra.utils.to_absolute_path('data/'), gender="neutral")
        self.joint_mapper = SMPLJointMapper()
        self.criterion_loss = torch.nn.L1Loss()
        self.feature_loss = torch.nn.MSELoss()
        self.weight_reg_criterion = torch.nn.MSELoss()

        self.domain_criterion = torch.nn.BCEWithLogitsLoss()

        self.confidence_threshold = 0.1
        self.confidence_dist = torch.distributions.Bernoulli(1 - self.occ_prob)

        self.optimizer = None
        self.critic_optimizer = None
        self.scheduler = None

        self.joint_weights = torch.ones(1, 18, 3).to("cuda")
        self.joint_weights[:, [4, 7, 10, 13]] = 8
        self.joint_weights[:, [3, 6, 9, 12]] = 4
        self.joint_weights[:, [2, 5, 8, 11]] = 2

    def get_model(self):
        # Constants
        NUM_JOINT = 18
        NUM_BETA = 10
        NUM_THETA = 24 * 6
        NUM_HIDDEN = self.train_opt.num_hidden
        DOMAIN_NUM_HIDDEN = self.train_opt.dom_num_hidden

        NUM_OUT = NUM_BETA + NUM_THETA
        NUM_IN = NUM_JOINT * 2

        feature_extractor_s = MLPFeatureExtractor(NUM_IN, NUM_HIDDEN)
        feature_extractor_t = None
        smpl_head = SMPLHead(NUM_HIDDEN, NUM_OUT, NUM_HIDDEN)
        domain_head = None

        if self.mode == "target":
            feature_extractor_t = MLPFeatureExtractor(NUM_IN, NUM_HIDDEN)
            domain_head = DomainCritic(NUM_HIDDEN, DOMAIN_NUM_HIDDEN, norm_layer="")

        return feature_extractor_s, feature_extractor_t, smpl_head, domain_head

    def smpl_head_forward(self, features, head='fs'):
        batch_size = features.shape[0]
        smpl_res = self.smpl_heads[head](features)
        poses_6d = smpl_res[:, 10:].reshape(batch_size * 24, 6)
        poses_rodrigues = geometry.poses_6d_to_rodrigues(poses_6d).reshape(-1, 24 * 3)

        res = {
            "betas": smpl_res[:, :10],
            "poses": poses_rodrigues
        }

        return res

    def forward(self, j2d):
        j2d_flatten = j2d.flatten(start_dim=1)
        features = self.feature_extractor(j2d_flatten)
        fs_smpl_head_res = self.smpl_head_forward(features, head='fs')
        return fs_smpl_head_res

    def configure_optimizers(self):
        params = chain(*[
            getattr(self, module).parameters()
            for module in self.training_modules
        ])
        self.optimizer = torch.optim.Adam(params, lr=self.train_opt.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=1,
                                                         gamma=0.8)
        self.critic_optimizer = None
        if self.mode == "target":
            self.critic_optimizer = torch.optim.Adam(self.domain_head.parameters(), lr=self.train_opt.critic_lr)

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """remove SMPL parameters"""
        keys = [key for key in checkpoint["state_dict"]
                if key.startswith("body_model")]
        for key in keys:
            del checkpoint["state_dict"][key]
        return super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """remove SMPL parameters"""
        keys = [key for key in checkpoint["state_dict"]
                if key.startswith("body_model")]
        for key in keys:
            del checkpoint["state_dict"][key]
        return super().on_save_checkpoint(checkpoint)

    def get_joints(self, thetas, betas):
        joint3d, smpl_output = self.get_joint3d(thetas, betas)
        joint2d = self.camera(joint3d)

        if self.train_opt.jitter_aug:
            joint2d += torch.from_numpy(
                np.round(np.random.multivariate_normal([0, 0], np.eye(2) * 1, (joint2d.shape[0], joint2d.shape[1]))),
            ).to(joint2d.device)

        confidence = self.confidence_dist.sample(
            joint2d.shape[:2]).to(self.device).float()
        normalized_joint2d = normalize_p2d(joint2d.clone(), confidence=confidence, threshold=self.confidence_threshold)

        return joint3d, joint2d, normalized_joint2d, confidence

    def training_step(self, batch, batch_idx, **kwargs):
        raise NotImplementedError

    def training_epoch_end(self, *args, **kwargs):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx, test=False):
        joint2d = batch['joint2d'].float()
        confidence = batch['confidence'].float()
        joint2d = normalize_p2d(joint2d, confidence=confidence, threshold=self.confidence_threshold)

        smpl_estimations = self.forward(joint2d)
        p = {
            "thetas": smpl_estimations['poses'],
            "betas": smpl_estimations['betas']
        }
        t = {
            "thetas": batch['poses'].float(),
            "betas": batch['betas'].float(),
            "joint3d": batch['joint3d'].float()
        }
        val_losses = self.compute_valid_errors(p, t)

        return {
            k: v for (k, v) in val_losses.items()
        }

    def validation_epoch_end(self, outputs):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, test=True)

    def test_epoch_end(self, outputs):
        total_loss_test = {}
        for output in outputs:
            for k, v in output.items():
                if k not in total_loss_test:
                    total_loss_test[k] = [v]
                else:
                    total_loss_test[k].append(v)
        res_to_w = {}
        for k, v in total_loss_test.items():
            res_to_w[k] = torch.cat(v).cpu().numpy()
            self.log(f"test/{k}", torch.cat(v).mean())

    def compute_valid_errors(self, x, y):
        joint3d_pr, smpl_out_pr = self.get_joint3d(x["thetas"], x["betas"], protocol='SPIN')
        joint3d_gt, smpl_out_gt = self.get_joint3d(y["thetas"], y["betas"], protocol='SPIN')
        verts_pr = smpl_out_pr.vertices
        gt_verts = smpl_out_gt.vertices
        pve_err = torch.sqrt(torch.sum((gt_verts - verts_pr) ** 2, dim=2)).mean(dim=1) * 1000
        err_spin_pa = compute_MPJPE(joint3d_pr, joint3d_gt * 1000)
        err_spin_mpjpe = compute_MPJPE(joint3d_pr * 1000, joint3d_gt * 1000, PA=False)

        return {
            "PA-MPJPE_SPIN": torch.from_numpy(np.asarray(err_spin_pa)),
            "MPJPE_SPIN": torch.from_numpy(np.asarray(err_spin_mpjpe)),
            "PVE": pve_err
        }

    def get_joint3d(self, theta, betas, protocol="COCO18", ret_smpl_out=False):
        smpl_output = self.body_model.forward(betas=betas,
                                              body_pose=theta[:, 3:],
                                              global_orient=theta[:, :3])

        joint3d = self.joint_mapper(smpl_output.joints,
                                    smpl_output.vertices,
                                    output_format=protocol)
        return joint3d, smpl_output
