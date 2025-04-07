#--------------------------------------------------------#
#   该文件用于调整输入彩色图片的后缀
#--------------------------------------------------------#
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

#--------------------------------------------------------#
#   Origin_JPEGImages_path   原始标签所在的路径
#   Out_JPEGImages_path      输出标签所在的路径
#--------------------------------------------------------#
# Origin_JPEGImages_path   = r"D:\OneDrive - The University of Auckland\IVSlab\timber_images\timber_images"
# Origin_JPEGImages_path = r"F:\dataset\RGB-D construction solid waste\Amanual\Amanual"
Origin_JPEGImages_path = r"F:\dataset\RGB-D construction solid waste\Aauto\Aauto\color"
Out_JPEGImages_path      = r"F:\dataset\CDW\All_color"


if __name__ == "__main__":
    if not os.path.exists(Out_JPEGImages_path):
        os.makedirs(Out_JPEGImages_path)

    #---------------------------#
    #   遍历标签并赋值
    #---------------------------#
    # image_names = os.listdir(Origin_JPEGImages_path)
    image_names = [f for f in os.listdir(Origin_JPEGImages_path) if f.endswith(('.jpg', '.png'))]
    print("正在遍历全部图片。")
    for image_name in tqdm(image_names):
        image   = Image.open(os.path.join(Origin_JPEGImages_path, image_name))
        image   = image.convert('RGB')
        image.save(os.path.join(Out_JPEGImages_path, os.path.splitext(image_name)[0] + '.jpg'))