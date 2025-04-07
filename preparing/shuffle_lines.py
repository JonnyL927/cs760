import random

# 输入和输出文件路径
input_file = r'D:\OneDrive - The University of Auckland\IVSlab\project\zeroWaste\UniMatch-V2\splits\CDW\5\5unlabeled.txt'
output_file = r'D:\OneDrive - The University of Auckland\IVSlab\project\zeroWaste\UniMatch-V2\splits\CDW\5\unlabeled.txt'

# 读取所有行并打乱顺序
with open(input_file, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

random.shuffle(lines)

# 将打乱后的行写入新文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    outfile.writelines(lines)

print(f"整行顺序已打乱，结果保存在 {output_file}")