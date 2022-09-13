import cv2
import os
import random

source_path = 'my_data/train/train10k'
save_path = 'my_data/train/train10k_blur'

def add_blur(img):
    # Add blur to image
    blur_type = random.randint(0, 2)
    blur_amount = random.randint(3, 10)
    if blur_amount % 2 == 0:
        blur_amount += 1
    if blur_type == 0:
        img = cv2.blur(img, (blur_amount, blur_amount))
    elif blur_type == 1:
        img = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)
    elif blur_type == 2:
        img = cv2.medianBlur(img, blur_amount)
    return img

for i, file in enumerate(os.listdir(source_path)):
    image = cv2.imread(f'{source_path}/{file}')
    image = add_blur(image)
    cv2.imwrite(f'{save_path}/{file}', image)
    length = len(os.listdir(source_path))
    if i < length-1:
        print(f'Bluring image {i}/{length-1}', end='\r')
    else:
        print(f'Bluring image {i}/{length-1}\nAllmages saved to {save_path}\n')
