import os
import cv2
import random
from tqdm import tqdm
import numpy as np

# 原始数据路径
# image_dir = r"F:\dataset\CDW\All" #F:\dataset\CDW\JPEGImages
# mask_dir = r"D:\OneDrive - The University of Auckland\IVSlab\timber_images\mask" #F:\dataset\CDW\SegmentationClass

image_dir = r"F:\dataset\CDW\5All"
mask_dir = r"F:\dataset\RGB-D construction solid waste\Amanual\Amanual\mask"



# # 输出目录
# output_dir = r"D:\OneDrive - The University of Auckland\IVSlab\project\zeroWaste\UniMatch-V2\splits\CDW\2"
# resize_dir = r"F:\dataset\CDW"

output_dir = r"D:\OneDrive - The University of Auckland\IVSlab\project\zeroWaste\UniMatch-V2\splits\CDW\5"
resize_dir = r"F:\dataset\CDW"


# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(resize_dir, "JPEGImages"), exist_ok=True)
os.makedirs(os.path.join(resize_dir, "SegmentationClass"), exist_ok=True)

# labeled 和 unlabeled 记录
labeled_path = os.path.join(output_dir, "labeled.txt")
unlabeled_path = os.path.join(output_dir, "unlabeled.txt")

# 获取所有图片
image_filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

# 存储 labeled 和 unlabeled 记录
labeled_entries = []
unlabeled_entries = []

for image_filename in tqdm(image_filenames, desc="Processing Images"):
    base_name, _ = os.path.splitext(image_filename)
    mask_filename = base_name + ".png"  # 统一使用 .png 作为掩码格式

    image_path = os.path.join(image_dir, image_filename)
    mask_path = os.path.join(mask_dir, mask_filename)

    resized_image_path = os.path.join(resize_dir, "JPEGImages", image_filename)
    resized_mask_path = os.path.join(resize_dir, "SegmentationClass", mask_filename)

    # 读取图像和掩码
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path) if os.path.exists(mask_path) else None

    if image is not None:
        # 统一调整尺寸
        image_resized = cv2.resize(image, (512, 512))  # , interpolation=cv2.INTER_LINEAR
        cv2.imwrite(resized_image_path, image_resized)  # 保存调整后的图片

        if mask is not None:
            # 转换为 RGB（以防 OpenCV 读取为 BGR）
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            # 初始化新的 mask（全为背景0）
            semantic_mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)

            # 标签颜色映射
            label_map = {
                (0, 128, 0): 1,  # wood/timber
                (0, 0, 128): 2,  # rubber
                (128, 0, 0): 3,  # brick
                (128, 128, 0): 4  # concrete
            }

            # 遍历每种颜色，标记对应类别
            for rgb, label in label_map.items():
                match = np.all(mask_rgb == rgb, axis=-1)
                semantic_mask[match] = label

            # 尺寸统一
            mask_resized = cv2.resize(semantic_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(resized_mask_path, mask_resized)  # 保存调整后的掩码

            labeled_entries.append(f"JPEGImages/{image_filename} SegmentationClass/{mask_filename}\n")
        else:
            unlabeled_entries.append(f"JPEGImages/{image_filename} SegmentationClass/{mask_filename}\n")
    #
    # if image is not None:
    #     # 统一调整尺寸
    #     image_resized = cv2.resize(image, (512, 512)) #, interpolation=cv2.INTER_LINEAR
    #     cv2.imwrite(resized_image_path, image_resized)  # 保存调整后的图片
    #
    #     if mask is not None:
    #         # 转换为 RGB（以防 OpenCV 读取为 BGR）
    #         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    #         # 生成二值 mask，背景=0，目标=1
    #         mask = np.all(mask == [128, 0, 0], axis=-1).astype(np.uint8)  # 目标为 1，背景为 0
    #         mask_resized = cv2.resize(mask, (512, 512)) #, interpolation=cv2.INTER_NEAREST
    #         cv2.imwrite(resized_mask_path, mask_resized)  # 保存调整后的掩码
    #
    #         labeled_entries.append(f"JPEGImages/{image_filename} SegmentationClass/{mask_filename}\n")
    #     else:
    #         unlabeled_entries.append(f"JPEGImages/{image_filename} SegmentationClass/{mask_filename}\n")

# 打乱顺序
random.shuffle(labeled_entries)
random.shuffle(unlabeled_entries)

# 写入文件
with open(labeled_path, 'w') as labeled_file:
    labeled_file.writelines(labeled_entries)
with open(unlabeled_path, 'w') as unlabeled_file:
    unlabeled_file.writelines(unlabeled_entries)

print("Processing complete. Files saved:")
print(f"- {labeled_path}")
print(f"- {unlabeled_path}")
print(f"- Resized images and masks saved in {resize_dir}")



# import os
# import cv2
# import random
# from tqdm import tqdm
#
# # 定义数据路径
# image_dir = r"F:\dataset\CDW\All" #JPEGImages
# mask_dir = r"F:\dataset\CDW\SegmentationClass"
# output_dir = r"D:\OneDrive - The University of Auckland\IVSlab\project\zeroWaste\UniMatch-V2\splits\CDW\2"
#
# # 确保输出目录存在
# os.makedirs(output_dir, exist_ok=True)
# labeled_path = os.path.join(output_dir, "labeled.txt")
# unlabeled_path = os.path.join(output_dir, "unlabeled.txt")
#
# # 获取所有图片文件（包括 .jpg 和 .png 格式）
# image_filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
#
# # 存储 labeled 和 unlabeled 记录
# labeled_entries = []
# unlabeled_entries = []
#
# for image_filename in tqdm(image_filenames, desc="Processing Images"):
#     image_rel_path = f"JPEGImages/{image_filename}"
#     base_name, _ = os.path.splitext(image_filename)
#     mask_filename = base_name + ".png"  # mask 统一用 .png 格式
#     mask_rel_path = f"SegmentationClass/{mask_filename}"
#     mask_path = os.path.join(mask_dir, mask_filename)
#
#     # 检查 mask 是否存在并可读取
#     if os.path.exists(mask_path) and cv2.imread(mask_path) is not None:
#         labeled_entries.append(f"{image_rel_path} {mask_rel_path}\n")
#     else:
#         unlabeled_entries.append(f"{image_rel_path} {mask_rel_path}\n")
#
# # 打乱顺序
# random.shuffle(labeled_entries)
# random.shuffle(unlabeled_entries)
#
# # 写入文件
# with open(labeled_path, 'w') as labeled_file:
#     labeled_file.writelines(labeled_entries)
# with open(unlabeled_path, 'w') as unlabeled_file:
#     unlabeled_file.writelines(unlabeled_entries)
#
# print("Processing complete. Files saved:")
# print(f"- {labeled_path}")
# print(f"- {unlabeled_path}")
