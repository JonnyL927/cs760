# Used under labelme environment Only


# import argparse
# import base64
# import json
# import os
# import os.path as osp
# import glob
#
# import imgviz
# import numpy as np
# from labelme import utils
# from labelme.logger import logger
#
#
# def collect_all_labels(json_dir):
#     label_set = set()
#     json_files = glob.glob(os.path.join(json_dir, '*.json'))
#     for json_file in json_files:
#         data = json.load(open(json_file))
#         for shape in data["shapes"]:
#             label_set.add(shape["label"])
#     label_list = sorted(list(label_set))
#     label_name_to_value = {"_background_": 0}
#     for i, name in enumerate(label_list):
#         label_name_to_value[name] = i + 1
#     return label_name_to_value
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("json_dir_file")
#     args = parser.parse_args()
#
#     json_dir = args.json_dir_file
#     json_files = glob.glob(os.path.join(json_dir, '*.json'))
#
#     # 统一类别映射
#     label_name_to_value = collect_all_labels(json_dir)
#
#     label_names = [None] * (max(label_name_to_value.values()) + 1)
#     for name, value in label_name_to_value.items():
#         label_names[value] = name
#
#     # 创建 mask 目录
#     mask_dir = osp.join(json_dir, 'mask')
#     os.makedirs(mask_dir, exist_ok=True)
#
#     for json_file in json_files:
#         data = json.load(open(json_file))
#         imageData = data.get("imageData")
#
#         if not imageData:
#             imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
#             with open(imagePath, "rb") as f:
#                 imageData = base64.b64encode(f.read()).decode("utf-8")
#
#         img = utils.img_b64_to_arr(imageData)
#         lbl, _ = utils.shapes_to_label(img.shape, data["shapes"], label_name_to_value)
#
#         mask_name = osp.splitext(osp.basename(json_file))[0] + ".png"
#         utils.lblsave(osp.join(mask_dir, mask_name), lbl)
#
#         logger.info(f"Saved mask: {mask_name}")
#
#     # 输出统一的 label_names.txt
#     with open(osp.join(json_dir, "label_names.txt"), "w") as f:
#         for lbl_name in label_names:
#             f.write(lbl_name + "\n")
#
#     logger.info(f"Saved label names: {osp.join(json_dir, 'label_names.txt')}")
#
#
# if __name__ == "__main__":
#     main()
