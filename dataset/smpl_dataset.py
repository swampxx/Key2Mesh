import hydra
import joblib
from torch.utils.data import Dataset

from utils.transforms import RandomRotateSMPL


class SingleFrameSMPLDataset(Dataset):
    def __init__(self, opt):
        self.db = joblib.load(hydra.utils.to_absolute_path(opt.dataset_path))
        if hasattr(opt, 'skip'):
            for k in self.db.keys():
                self.db[k] = self.db[k][::opt.skip]

        if opt.augmentation:
            self.transform = RandomRotateSMPL()
        else:
            self.transform = None

    def __len__(self):
        return self.db['theta'].shape[0]

    def __getitem__(self, item):
        thetas = self.db['theta'][item]

        betas = thetas[:10]
        poses = thetas[10:]

        res = {
            "poses": poses,
            "betas": betas
        }

        if self.transform:
            res = self.transform(res)
        return res


class SingleFrame3DPWDataset(SingleFrameSMPLDataset):
    def __init__(self, opt):
        super(SingleFrame3DPWDataset, self).__init__(opt)

    def __len__(self):
        return self.db['pose'].shape[0]

    def __getitem__(self, item):
        res = {
            "poses": self.db['pose'][item],
            "betas": self.db['shape'][item],
            "joint2d": self.db['joints2D'][item][:, :2],
            "joint3d": self.db['joints3D'][item] * 1000,
            "confidence": self.db['joints2D'][item][:, 2],
            "img_name": self.db["img_name"][item]
        }

        if self.transform:
            res = self.transform(res)
        return res
