from torch.utils.data import DataLoader
from torchvision import transforms
from .data_transform import ProcessImages


def setup_dada2ks(DADA2KS, cfg, num_workers=0, isTraining=True):
    """Both mean and std are arranged by the RGB order
    """
    transform_dict = {'image': transforms.Compose([ProcessImages(cfg.input_shape, mean=[0.218, 0.220, 0.209], std=[0.277, 0.280, 0.277])])}
    traindata_loader, evaldata_loader, testdata_loader = None, None, None
    # testing dataset
    if isTraining:
        # training dataset
        train_data = DADA2KS(cfg.data_path, 'training', interval=cfg.frame_interval, transforms=transform_dict, data_aug=cfg.data_aug)
        traindata_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
        cfg.update(FPS=train_data.fps)
        # validataion dataset
        eval_data = DADA2KS(cfg.data_path, 'validation', interval=cfg.frame_interval, transforms=transform_dict, data_aug=cfg.data_aug)
        evaldata_loader = DataLoader(dataset=eval_data, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)
        print("# train set: %d, eval set: %d"%(len(train_data), len(eval_data)))
    else:
        test_data = DADA2KS(cfg.data_path, 'testing', interval=cfg.frame_interval, transforms=transform_dict)
        testdata_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)
        cfg.update(FPS=test_data.fps)
        print("# test set: %d"%(len(test_data)))
    return traindata_loader, evaldata_loader, testdata_loader


def setup_dad(DADDataset, cfg, num_workers=0, isTraining=True):
    transform = transforms.Compose([ProcessImages(cfg.input_shape, mean=[0.315, 0.316, 0.318], std=[0.294, 0.299, 0.303])])
    traindata_loader, evaldata_loader, testdata_loader = None, None, None
    # testing dataset
    if isTraining:
        # training dataset
        train_data = DADDataset(cfg.data_path, 'train', interval=cfg.frame_interval, transforms=transform)
        traindata_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
        cfg.update(FPS=train_data.fps)
        # validataion dataset
        eval_data = DADDataset(cfg.data_path, 'val', interval=cfg.frame_interval, transforms=transform)
        evaldata_loader = DataLoader(dataset=eval_data, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)
        print("# train set: %d, eval set: %d"%(len(train_data), len(eval_data)))
    else:
        test_data = DADDataset(cfg.data_path, 'test', interval=cfg.frame_interval, transforms=transform)
        testdata_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)
        cfg.update(FPS=test_data.fps)
        print("# test set: %d"%(len(test_data)))
    return traindata_loader, evaldata_loader, testdata_loader