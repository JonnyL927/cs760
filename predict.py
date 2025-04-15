import torch
import torch.nn.functional as F
import yaml
import os
import cv2
import numpy as np
from PIL import Image
from model.semseg.dpt import DPT
from util.classes import CLASSES
from torchvision import transforms

# PALETTE = [
#     [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156], [190,153,153],
#     [153,153,153], [250,170, 30], [220,220,  0], [107,142, 35], [152,251,152],
#     [ 70,130,180], [220, 20, 60], [255,  0,  0], [  0,  0,142], [  0,  0, 70],
#     [  0, 60,100], [  0, 80,100], [  0,  0,230], [119, 11, 32]
# ]
PALETTE = [
    [0, 0, 0],
    [128, 0, 0], #timber red
    [0, 128, 0], #concrete green
    [0, 0, 128], #brick blue
    [128, 128, 0] #rubber yellow
]

def colorize_mask(mask, palette):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_mask[mask == label] = color
    return color_mask


def load_model(cfg, model_path, numberOfClass):
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    backbone_type = cfg['backbone'].split('_')[-1]
    model = DPT(**{**model_configs[backbone_type], 'nclass': numberOfClass})
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval().cuda()
    return model


def preprocess_image(img_path, img_size):
    img = Image.open(img_path).convert('RGB')
    ori_size = img.size  # (W, H)
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    tensor = transform(img).unsqueeze(0)  # (1,3,H,W)
    return tensor.cuda(), ori_size, np.array(img)


@torch.no_grad()
def predict(model, img_tensor, ori_size, palette):
    pred = model(img_tensor)
    pred = F.interpolate(pred, size=ori_size[::-1], mode='bilinear', align_corners=True)
    pred_mask = pred.argmax(1).squeeze().cpu().numpy()
    color_mask = colorize_mask(pred_mask, palette)
    return color_mask


if __name__ == '__main__':
    config_path = 'configs/CDW.yaml'
    model_path = 'exp/CDW/unimatch_v2/output/dinov2_small_CDW/5_0.2/best.pth'
    img_path = 'F:/dataset/CDW/JPEGImages_full/202004010213.jpg'
    # img_path = r'F:\dataset\RGB-D construction solid waste\Amanual\Amanual\202003300155.png'
    save_path = 'exp/CDW/unimatch_v2/predict/photos/202004010213.2.png'
    numberOfClass = 5
    image_size = [518, 518]

    cfg = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    cfg.setdefault('image_size', image_size)

    model = load_model(cfg, model_path, numberOfClass)
    img_tensor, ori_size, raw_img = preprocess_image(img_path, cfg['image_size'])
    color_mask = predict(model, img_tensor, ori_size, PALETTE)

    # ->BRG
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(raw_img, 0.2, color_mask, 0.8, 0)
    cv2.imwrite(save_path, overlay)
    print(f"Saved prediction to {save_path}")






# python predict.py --config=configs/CDW.yaml --model-path exp/CDW/unimatch_v2/output/dinov2_small_CDW/5/best.pth --img-path F:/dataset/CDW/JPEGImages/76.jpg --save-path exp/CDW/unimatch_v2/predict/photos/output.png