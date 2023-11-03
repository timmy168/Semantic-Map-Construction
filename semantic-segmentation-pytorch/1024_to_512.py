import os
from PIL import Image

# 源图像文件夹和目标图像文件夹
source_folder = "data/first_floor/images/result(model_1)"
target_folder = "data/first_floor/images/result_512(model_1)"

# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 循环处理源文件夹中的图像
for filename in os.listdir(source_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # 打开源图像
        source_path = os.path.join(source_folder, filename)
        img = Image.open(source_path)

        # 获取图像的宽度和高度
        width, height = img.size

        # 裁剪图像的右半部分（512x512）
        left = width - 512
        upper = 0
        right = width
        lower = 512
        img_cropped = img.crop((left, upper, right, lower))

        # 生成目标文件名（可根据需要进行更改）
        target_filename = filename

        # 保存裁剪后的图像到目标文件夹
        target_path = os.path.join(target_folder, target_filename)
        img_cropped.save(target_path)

        print(f"Cropped and saved {target_filename}")

print("All images processed.")
