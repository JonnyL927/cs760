import torch.distributed as dist
import torch.nn.functional as F

from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
import argparse
import logging
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, RandomSampler
import yaml

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.utils import count_params, init_log, AverageMeter
import numpy as np





parser = argparse.ArgumentParser(description='UniMatch V2 Testing: Evaluate Model on Validation Set')
parser.add_argument('--config', type=str, required=True, help='Path to config file')
parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
parser.add_argument('--save-path', type=str, required=True, help='Path to save results')


# def evaluate(model, loader, cfg):
#     model.eval()
#     intersection_meter = AverageMeter()
#     union_meter = AverageMeter()
#
#     with torch.no_grad():
#         for img, mask, _ in loader:
#             img, mask = img.cuda(), mask.cuda()
#             pred = model(img)
#             pred = pred.argmax(dim=1)
#
#             intersection, union, _ = intersectionAndUnion(pred.cpu().numpy(), mask.cpu().numpy(), cfg['nclass'], 255)
#             intersection_meter.update(intersection)
#             union_meter.update(union)
#
#     iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
#     mean_iou = iou_class.mean()
#     return mean_iou, iou_class
def evaluate(model, loader, mode, cfg, multiplier=None):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, id in loader:

            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()

                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: row + grid, col: col + grid])
                        final[:, :, row: row + grid, col: col + grid] += pred.softmax(dim=1)
                        if col == w - grid:
                            break
                        col = min(col + int(grid * 2 / 3), w - grid)
                    if row == h - grid:
                        break
                    row = min(row + int(grid * 2 / 3), h - grid)

                pred = final

            else:
                assert mode == 'original'

                if multiplier is not None:
                    ori_h, ori_w = img.shape[-2:]
                    if multiplier == 512:
                        new_h, new_w = 512, 512
                    else:
                        new_h, new_w = int(ori_h / multiplier + 0.5) * multiplier, int(
                            ori_w / multiplier + 0.5) * multiplier
                    img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True)

                pred = model(img)

                if multiplier is not None:
                    pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)

            pred = pred.argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            if dist.is_initialized():
                dist.all_reduce(reduced_intersection)
                dist.all_reduce(reduced_union)
                dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = init_log('test', logging.INFO)
    os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    model.cuda()

    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    logger.info('Loaded model from {}'.format(args.model_path))

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    mean_iou, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)

    logger.info('Mean IoU: {:.2f}'.format(mean_iou))
    for i, iou in enumerate(iou_class):
        logger.info('Class [{}] IoU: {:.2f}'.format(i, iou))

    with open(os.path.join(args.save_path, 'test_results5.txt'), 'w') as f:
        f.write('Mean IoU: {:.2f}\n'.format(mean_iou))
        for i, iou in enumerate(iou_class):
            f.write('Class [{}] IoU: {:.2f}\n'.format(i, iou))

    logger.info('Evaluation results saved to {}'.format(args.save_path))


if __name__ == '__main__':
    main()

#python test.py --config configs/CDW.yaml --model-path exp/CDW/unimatch_v2/output/dinov2_small_CDW/2/best.pth --save-path exp/CDW/unimatch_v2/predict
