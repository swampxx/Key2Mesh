from torch.utils.data import DataLoader

from .smpl_dataset import  SingleFrameSMPLDataset, SingleFrame3DPWDataset


def find_dataset_using_name(name):
    mapping = {
        "SingleFrame3DPWDataset": SingleFrame3DPWDataset,
        "SingleFrameSMPLDataset": SingleFrameSMPLDataset,
    }
    cls = mapping.get(name, None)
    if cls is None:
        raise ValueError(f"Fail to find dataset {name}")
    return cls


def create_dataset(opt):
    dataset_cls = find_dataset_using_name(opt.name)
    dataset = dataset_cls(opt)
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        drop_last=opt.drop_last,
        shuffle=opt.shuffle,
        num_workers=opt.worker,
        pin_memory=True
    )
