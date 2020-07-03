import numpy as np
import pandas as pd
import cv2
import os
import sys
import ast
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import uuid


def xyxy2xywh(x):
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def normalize(lab, siz):
    lab[:, 1:5] = xyxy2xywh(lab[:, 1:5])
    lab[:, [2, 4]] /= siz[1]
    lab[:, [1, 3]] /= siz[0]
    return lab


def no_resize_and_no_pad(img_path):
    img = cv2.imread(img_path)
    H, W, _ = img.shape
    wide = False if H > W else True
    return img, 0, wide, 1


def resize_and_pad(img_path, op_size):
    img = cv2.imread(img_path)
    H, W, _ = img.shape
    canvas = np.zeros((op_size, op_size, 3))
    for i in range(3):
        canvas[:, :, i] = np.mean(img[:, :, i])
    if H > W:
        new_H = op_size
        new_W = int(float(op_size * W) / H)
        ratio = float(op_size) / H
        d = new_H - new_W
        wide = False
    else:
        new_W = op_size
        new_H = int(float(op_size * H) / W)
        ratio = float(op_size) / W
        d = new_W - new_H
        wide = True
    dl = random.randint(0, d)
    I_res = cv2.resize(img, (new_W, new_H))
    if H > W:
        canvas[:, dl:dl + new_W, :] = I_res
    else:
        canvas[dl:dl + new_H, :, :] = I_res
    return canvas, dl, wide, ratio


def normalized_format(boxes, dl, wide, ratio, op_size_x, op_size_y):
    out = []
    for v in boxes:
        left, top, right, bottom = v
        if wide:
            top_new = int(top * ratio) + dl
            bottom_new = int(bottom * ratio) + dl
            left_new = int(left * ratio)
            right_new = int(right * ratio)
        else:
            top_new = int(top * ratio)
            bottom_new = int(bottom * ratio)
            left_new = int(left * ratio) + dl
            right_new = int(right * ratio) + dl
        x_new = (abs(float((left_new + right_new)))) / (2 * op_size_x)
        y_new = (abs(float((top_new + bottom_new)))) / (2 * op_size_y)
        w_new = (abs(float((right_new - left_new)))) / op_size_x
        h_new = (abs(float((bottom_new - top_new)))) / op_size_y
        out.append([x_new, y_new, w_new, h_new])
    return out


def resize_img_labels(img_path, lis, out_path, op_size):
    crops = [n['labels_crops'] for n in lis]
    if op_size:
        canvas, dl, wide, ratio = resize_and_pad(img_path, op_size)
        norm_crop = normalized_format(crops, dl, wide, ratio, op_size, op_size)
    else:
        canvas, dl, wide, ratio = no_resize_and_no_pad(img_path)
        H, W, _ = canvas.shape
        norm_crop = normalized_format(crops, dl, wide, ratio, W, H)
    filename = str(uuid.uuid5(uuid.NAMESPACE_URL, img_path)) + '.jpg'
    out_image = os.path.join(out_path, "images", filename)
    cv2.imwrite(out_image, canvas)
    out_txt = os.path.join(out_path, "labels", '.'.join(filename.split('.')[:-1]) + '.txt')
    lab_str = '\n'.join([' '.join([str(lis[i]['labels_idx'])] + [str(m) for m in n]) for i, n in enumerate(norm_crop)])
    fd = open(out_txt, 'w')
    fd.write(lab_str)
    fd.close()
    return lis


def save_data_parallel(dic, out_path, op_size, no_process=50):
    # async download
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = asyncio.ensure_future(async_save_data(dic, out_path, op_size, no_process))
    loop.run_until_complete(future)
    print("Image Download Done")


async def async_save_data(dic, out_path, op_size, no_process):
    # Starting down
    with ThreadPoolExecutor(max_workers=no_process) as executor:
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                executor,
                resize_img_labels,
                k, v,
                out_path,
                op_size
            )
            for k, v in dic.items()
        ]
        responses = []
        for f in tqdm(asyncio.as_completed(futures), total=len(futures), disable=False):
            responses.append(await f)
        for response in await asyncio.gather(*futures):
            pass


def save_data(dic, out_path, op_size):
    _ = {k: resize_img_labels(k, v, out_path, op_size) for k, v in tqdm(dic.items())}


def prep_yolo(df, classes, out_path, op_size, txt_file_name="img.txt"):
    if not os.path.isdir(os.path.join(out_path, 'images')):
        os.makedirs(os.path.join(out_path, 'images'))
    if not os.path.isdir(os.path.join(out_path, 'labels')):
        os.makedirs(os.path.join(out_path, 'labels'))
    # format
    df['labels_idx'] = df.labels.apply(lambda x: classes.index(x))
    df['labels_crops'] = df.labels_crops.apply(lambda x: ast.literal_eval(x))
    # combine
    dic = {path: df_sub.to_dict(orient='records') for path, df_sub in tqdm(df.groupby('img_path'))}
    # save_data(dic, out_path, op_size)
    save_data_parallel(dic, out_path, op_size)
    img_lis = os.listdir(os.path.join(out_path, 'images'))
    images = '\n'.join([os.path.join(out_path, 'images', n) for n in img_lis])
    with open(os.path.join(out_path, txt_file_name), 'w') as d:
        d.write(images)

    path_dic = {str(uuid.uuid5(uuid.NAMESPACE_URL, path)) + '.jpg': path for path in df.img_path.unique().tolist()}

    return path_dic


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    classes = sys.argv[2].split(',')
    out_path = sys.argv[3]
    op_size = sys.argv[4] if sys.argv[4].lower() not in ["none", "nan", "false"] else None
    prep_yolo(df, classes, out_path, op_size)
