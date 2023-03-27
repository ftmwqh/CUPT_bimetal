import numpy as np
import cv2
import os
import imageio
from PIL import Image
def update(input_img_path, output_img_path):
    image = cv2.imread(input_img_path)
    print(image.shape)
    cropped = image[200:245, 382:405]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite(output_img_path, cropped)


dataset_dir = '55C30W'
output_dir = '55C30W_1_2'

# 获得需要转化的图片路径并生成目标路径
image_filenames = [(os.path.join(dataset_dir, x), os.path.join(output_dir, x))
                   for x in os.listdir(dataset_dir)]
# 转化所有图片
for path in image_filenames:
    update(path[0], path[1])

def read_file_pics(path_name):
    path=path_name
    path_list=os.listdir(path)

    path_list.sort(key=lambda x:int(x[:-4]))
    print(path_list)
    for filename in path_list:
        img_array = imageio.imread(os.path.join(path,filename), as_gray=True)
        img_array = 0 + (img_array > 150) * 254
        print(img_array)
        pic = Image.fromarray(img_array)
        pic.show()


read_file_pics(r'E:\py_projects\bimetal_vib\55C30W_1_1')#不转义加r或R