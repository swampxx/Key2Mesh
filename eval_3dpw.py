import glob
import os

import hydra
import pytorch_lightning as pl
import torch

from dataset import create_dataset
from models.key2mesh import Key2Mesh


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, testset):
        super().__init__()
        self.testset = testset

    def test_dataloader(self):
        return self.testset


@hydra.main(config_path="configs", config_name="eval_3dpw")
def main(opt):
    pl.seed_everything(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    testset = create_dataset(opt.data.test)
    datamodule = CustomDataModule(testset)

    checkpoints = sorted(glob.glob("ckpts/best*.ckpt"), key=os.path.getmtime)
    model = Key2Mesh.load_from_checkpoint(checkpoints[-1], strict=False, opt=opt.model,
                                          loss_opt=opt.loss,
                                          train_opt=opt.train)
    model.train()
    trainer = pl.Trainer(gpus=1, accelerator="gpu")
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
