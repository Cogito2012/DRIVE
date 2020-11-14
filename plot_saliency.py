import os, cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.saliency.mlnet import MLNet
import torch
from torchvision import transforms
from src.data_transform import ProcessImages, ProcessFixations
# from src.TorchFovea import TorchFovea


def minmax_norm(salmap):
    """Normalize the saliency map with min-max
    salmap: (B, 1, H, W)
    """
    batch_size, height, width = salmap.size(0), salmap.size(2), salmap.size(3)
    salmap_data = salmap.view(batch_size, -1)  # (B, H*W)
    min_vals = salmap_data.min(1, keepdim=True)[0]  # (B, 1)
    max_vals = salmap_data.max(1, keepdim=True)[0]  # (B, 1)
    salmap_norm = (salmap_data - min_vals) / (max_vals - min_vals)
    salmap_norm = salmap_norm.view(batch_size, 1, height, width)
    return salmap_norm


def saliency_padding(saliency, image_size):
    """Up padding the saliency (B, 60, 80) to image size (B, 330, 792)
    """
    # get size and ratios
    height, width = saliency.shape[1:]
    rows_rate = image_size[0] / height  # h ratio (5.5)
    cols_rate = image_size[1] / width   # w ratio (9.9)
    # padding
    if rows_rate > cols_rate:
        pass
    else:
        new_rows = (image_size[0] * width) // image_size[1]
        patch_ctr = saliency[:, ((height - new_rows) // 2):((height - new_rows) // 2 + new_rows), :]
        patch_ctr = np.rollaxis(patch_ctr, 0, 3)
        padded = cv2.resize(patch_ctr, (image_size[1], image_size[0]))
        padded = np.rollaxis(padded, 2, 0)
    return padded


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Visualize Results')
    # For training and testing
    parser.add_argument('--sal_ckpt', default='models/saliency/mlnet_25.pth',
                        help='Pretrained model for bottom-up saliency prediciton.')
    parser.add_argument('--imgs_path', default='examples/')
    parser.add_argument('--output', default='output/',
                        help='Directory of the output. ')
    args = parser.parse_args()
    image_size = [660, 1584]
    height, width = 480, 640

    # prepare output directory
    output_dir = os.path.join(args.output, 'vis_salmaps')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # environmental model
    device = torch.device("cuda:0")
    observe_model = MLNet((height, width))
    assert os.path.exists(args.sal_ckpt), "Checkpoint directory does not exist! %s"%(args.sal_ckpt)
    ckpt = torch.load(args.sal_ckpt, map_location=device)
    observe_model.load_state_dict(ckpt['model'])
    observe_model.to(device)
    observe_model.eval()
    # fovealmodel = TorchFovea((height, width), min(height, width)/6.0, level=5, factor=2, device=device)
    # transform
    data_trans = transforms.Compose([ProcessImages((height, width), mean=[0.218, 0.220, 0.209], std=[0.277, 0.280, 0.277])])

    frame_data, names = [], []
    for filename in os.listdir(args.imgs_path):
        frame = cv2.imread(os.path.join(args.imgs_path, filename))
        frame_data.append(frame)
        names.append(filename)

    frame_data = np.array(frame_data)
    input_data = torch.FloatTensor(data_trans(frame_data)).to(device)
    with torch.no_grad():
        saliency = observe_model(input_data)  # (B, 1, 60, 80)
        saliency = minmax_norm(saliency)
        salmap = saliency.squeeze(1).cpu().numpy()
    salmap = saliency_padding(salmap, image_size)

    for i, (filename, frame) in enumerate(zip(names, frame_data)):
        heatmap = cv2.applyColorMap((salmap[i] * 255).astype(np.uint8), cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

        result_file = os.path.join(output_dir, filename[:-4] + '_salmap.png')
        cv2.imwrite(result_file, frame)