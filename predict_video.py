import cv2
import torch
import torch.nn.functional as F
import yaml
import numpy as np
from PIL import Image
from model.semseg.dpt import DPT
from torchvision import transforms
import os

PALETTE = [
    [0, 0, 0],
    [128, 0, 0],    #timber     red
    [0, 128, 0],    #concrete   green
    [0, 0, 128],    #brick      blue
    [128, 128, 0]   #rubber     yellow
]

def colorize_mask(mask, palette):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_mask[mask == label] = color
    return color_mask

def load_model(cfg, model_path, nclass):
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    backbone_type = cfg['backbone'].split('_')[-1]
    model = DPT(**{**model_configs[backbone_type], 'nclass': nclass})
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval().cuda()
    return model

def preprocess_frame(frame, img_size):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ori_size = img.size
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    tensor = transform(img).unsqueeze(0).cuda()
    return tensor, ori_size

@torch.no_grad()
def predict(model, tensor, ori_size, palette):
    pred = model(tensor)
    pred = F.interpolate(pred, size=ori_size[::-1], mode='bilinear', align_corners=True)
    pred_mask = pred.argmax(1).squeeze().cpu().numpy()
    color_mask = colorize_mask(pred_mask, palette)
    return color_mask

if __name__ == '__main__':
    config_path = 'configs/CDW.yaml'
    model_path = 'exp/CDW/unimatch_v2/output/dinov2_small_CDW/5_0.2/best.pth'
    video_path = r'D:\OneDrive - The University of Auckland\IVSlab\timber_video\timber_video\IMG_3165.MP4.MP4'
    save_path = 'exp/CDW/unimatch_v2/predict/videos/output_best600.2.mp4'
    numberOfClass = 5
    image_size = [518, 518]

    # loading model
    cfg = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    cfg.setdefault('image_size', image_size)
    model = load_model(cfg, model_path, numberOfClass)

    # open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # output format
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tensor, ori_size = preprocess_frame(frame, image_size)
        color_mask = predict(model, tensor, ori_size, PALETTE)
        color_mask = cv2.resize(color_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(frame, 0.2, color_mask, 0.8, 0)

        writer.write(overlay)
        frame_count += 1
        print(f"Processed frame {frame_count}", end='\r')

    cap.release()
    writer.release()
    print(f"Video saved to {save_path}")
