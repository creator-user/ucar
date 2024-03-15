import glob
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.utils import xyxy2xywh



class LoadImages:  # for inference
    def __init__(self, path, img_size=416):
        self.height = img_size
        img_formats = ['.jpg', '.jpeg', '.png', '.tif']
        vid_formats = ['.mov', '.avi', '.mp4']

        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob('%s/*.*' % path))
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)

        self.files = images + videos
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in ' + path

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'File Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height)
        print('%gx%g ' % img.shape[:2], end='')  # print image size

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files

# # -------------------------------原始代码--------------------------------
# # 定义了一个用于从摄像头读取图像并进行预处理的类。它通过迭代器协议允许您从摄像头连续读取图像，并将图像调整为指定的尺寸。
# # 如果按下"Esc"键，程序会关闭OpenCV窗口并终止迭代。注意，__len__ 方法返回0，这意味着在这个类中没有预定义的数据数量。
# class LoadWebcam:  # 用于推理的类
#     def __init__(self, img_size=416):
#         self.cam = cv2.VideoCapture(0)  # 打开摄像头
#         self.height = img_size  # 图像尺寸

#     def __iter__(self):
#         self.count = -1
#         return self

#     def __next__(self):
#         self.count += 1
#         if cv2.waitKey(1) == 27:  # 按下"Esc"键退出
#             cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
#             raise StopIteration

#         # 读取图像
#         ret_val, img0 = self.cam.read()
#         assert ret_val, 'Webcam Error'  # 断言，如果读取图像失败则抛出异常
#         img_path = 'webcam_%g.jpg' % self.count
#         img0 = cv2.flip(img0, 1)  # 左右翻转图像

#         # 填充并调整大小
#         img, _, _, _ = letterbox(img0, height=self.height)

#         # 归一化RGB值
#         img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR转RGB
#         img = np.ascontiguousarray(img, dtype=np.float32)  # uint8转float32
#         img /= 255.0  # 将像素值从0 - 255缩放到0.0 - 1.0之间

#         return img_path, img, img0, None

#     def __len__(self):
#         return 0  # 返回长度为0，表示没有预定义的数据数量
# # -------------------------------原始代码--------------------------------


#------------------------------色温迁移-----------------------------
class LoadWebcam:
    def __init__(self, img_size=416):
        self.cam = cv2.VideoCapture(0)  # 打开摄像头
        self.height = img_size  # 图像尺寸
        self.source_image = cv2.imread('source.jpg')

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == 27:  # 按下"Esc"键退出
            cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
            raise StopIteration

        ret_val, img0 = self.cam.read()
        assert ret_val, 'Webcam Error'  # 断言，如果读取图像失败则抛出异常
        img_path = 'webcam_%g.jpg' % self.count
        img0 = cv2.flip(img0, 1)  # 左右翻转图像

        # 进行色温迁移
        if self.source_image is not None:
            img0 = self.color_temperature_shift(img0, self.source_image)

        # 填充并调整大小
        # letterbox函数返回的是一个元组 (img, ratio, dw, dh)
        img, _, _, _ = letterbox(img0, height=self.height)

        # 归一化RGB值
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0, None

    def __len__(self):
        return 0  # 返回长度为0，表示没有预定义的数据数量

    def calculate_color_temperature(self, image):
        # 在这里可以通过计算像素的平均颜色或直方图分析来获取色温信息
        # 这里仅作示例，使用简单的均值计算
        mean_color = np.mean(image, axis=(0, 1))
        return mean_color

    def color_temperature_shift(self, source_image, target_image):
        # 获取参考图片和目标图片的色温
        source_temperature = self.calculate_color_temperature(source_image)
        target_temperature = self.calculate_color_temperature(target_image)

        # 计算色温差异
        temperature_difference = target_temperature - source_temperature

        # 色温迁移
        result_image = np.clip(source_image + temperature_difference, 0, 255).astype(np.uint8)

        return result_image
#------------------------------色温迁移-----------------------------



# #------------------------------色温和亮度迁移-----------------------------
# class LoadWebcam:
#     def __init__(self, img_size=416):
#         self.cam = cv2.VideoCapture(0)  # 打开摄像头
#         self.height = img_size  # 图像尺寸
#         self.source_image = cv2.imread('source.jpg')

#     def __iter__(self):
#         self.count = -1
#         return self

#     def __next__(self):
#         self.count += 1
#         if cv2.waitKey(1) == 27:  # 按下"Esc"键退出
#             cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
#             self.cam.release()  # 释放摄像头资源
#             raise StopIteration

#         ret_val, img0 = self.cam.read()
#         assert ret_val, 'Webcam Error'  # 断言，如果读取图像失败则抛出异常
#         img_path = 'webcam_%g.jpg' % self.count
#         img0 = cv2.flip(img0, 1)  # 左右翻转图像

#         # 进行色温迁移
#         if self.source_image is not None:
#             img0 = self.color_temperature_shift(img0, self.source_image)

#         # 进行亮度迁移
#         if self.source_image is not None:
#             img0 = self.transfer_brightness(self.source_image, img0)

#         # 填充并调整大小
#         img, _, _, _ = letterbox(img0, height=self.height)

#         # 归一化RGB值
#         img = img[:, :, ::-1].transpose(2, 0, 1)
#         img = np.ascontiguousarray(img, dtype=np.float32)
#         img /= 255.0

#         return img_path, img, img0, None

#     def __len__(self):
#         return 0  # 返回长度为0，表示没有预定义的数据数量
    
#     def calculate_color_temperature(self, image):
#         # 在这里可以通过计算像素的平均颜色或直方图分析来获取色温信息
#         # 这里仅作示例，使用简单的均值计算
#         mean_color = np.mean(image, axis=(0, 1))
#         return mean_color

#     def color_temperature_shift(self, source_image, target_image):
#         # 获取参考图片和目标图片的色温
#         source_temperature = self.calculate_color_temperature(source_image)
#         target_temperature = self.calculate_color_temperature(target_image)

#         # 计算色温差异
#         temperature_difference = target_temperature - source_temperature

#         # 色温迁移
#         result_image = np.clip(source_image + temperature_difference, 0, 255).astype(np.uint8)

#         return result_image

#     def transfer_brightness(self, source_img, target_img):
#         # 转换为Lab色彩空间
#         source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2Lab)
#         target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2Lab)

#         # 提取亮度信息（L通道）
#         source_l_channel = source_lab[:,:,0]
#         target_l_channel = target_lab[:,:,0]

#         # 计算亮度差异
#         source_mean = np.mean(source_l_channel)
#         target_mean = np.mean(target_l_channel)
#         delta_l = target_mean - source_mean

#         # 调整目标图片的亮度（应用亮度迁移）
#         adjusted_l_channel = np.clip(target_l_channel - delta_l, 0, 255).astype(np.uint8)

#         # 重新组合Lab通道
#         transfer_lab = target_lab.copy()
#         transfer_lab[:,:,0] = adjusted_l_channel

#         # 转换回RGB色彩空间
#         transfer_bgr = cv2.cvtColor(transfer_lab, cv2.COLOR_Lab2BGR)

#         return transfer_bgr
# #------------------------------色温和亮度迁移-----------------------------


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=416, augment=False):
#         print('---------------------------', os.getcwd(), '------------------------------')
        with open(path, 'r') as f:
            img_files = f.read().splitlines()
            self.img_files = list(filter(lambda x: len(x) > 0, img_files))

        n = len(self.img_files)
        assert n > 0, 'No images found in %s' % path
        self.img_size = img_size
        self.augment = augment
        self.label_files = [
            x.replace('images', 'labels').replace('.bmp', '.txt').replace('.jpg', '.txt').replace('.png', '.txt')
            for x in self.img_files]

        # sort dataset by aspect ratio for rectangular training
        self.rectangle = False
        if self.rectangle:
            from PIL import Image

            s = np.array([Image.open(f).size for f in tqdm(self.img_files, desc='Reading image shapes')])
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            i = ar.argsort()
            self.img_files = [self.img_files[i] for i in i]
            self.label_files = [self.label_files[i] for i in i]
            self.ar = ar[i]

        # if n < 200:  # preload all images into memory if possible
        #    self.imgs = [cv2.imread(img_files[i]) for i in range(n)]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        label_path = self.label_files[index]
#         print('---------------------------', img_path, '------------------------------')
#         print('---------------------------', label_path, '------------------------------')
        
#         print('---------------------------', os.getcwd(), '------------------------------')
#         print('---------------------------', label_path, '------------------------------')
        # if hasattr(self, 'imgs'):
        #    img = self.imgs[index]  # BGR
        img = cv2.imread(img_path)  # BGR
        assert img is not None, 'File Not Found ' + img_path

        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50  # must be < 1.0
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
            S = img_hsv[:, :, 1].astype(np.float32)  # saturation
            V = img_hsv[:, :, 2].astype(np.float32)  # value

            a = (random.random() * 2 - 1) * fraction + 1
            b = (random.random() * 2 - 1) * fraction + 1
            S *= a
            V *= b

            img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=self.img_size, mode='square')

        # Load labels
        labels = []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as file:
                lines = file.read().splitlines()

            x = np.array([x.split() for x in lines], dtype=np.float32)
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio * w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = ratio * h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = ratio * w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = ratio * h * (x[:, 2] + x[:, 4] / 2) + padh

        # Augment image and labels
        if self.augment:
            img, labels = random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5]) / self.img_size

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() > 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() > 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return torch.from_numpy(img), labels_out, img_path, (h, w)

    @staticmethod
    def collate_fn(batch):
        img, label, path, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, hw


def letterbox(img, height=416, color=(127.5, 127.5, 127.5), mode='rect'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]

    # Select padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'rect':  # rectangle
        dw = np.mod(height - new_shape[0], 32) / 2  # width padding
        dh = np.mod(height - new_shape[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (height - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


def random_affine(img, targets=(), degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if len(targets) > 0:
        n = targets.shape[0]
        points = targets[:, 1:5].copy()
        area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # apply angle-based reduction of bounding boxes
        radians = a * math.pi / 180
        reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        x = (xy[:, 2] + xy[:, 0]) / 2
        y = (xy[:, 3] + xy[:, 1]) / 2
        w = (xy[:, 2] - xy[:, 0]) * reduction
        h = (xy[:, 3] - xy[:, 1]) * reduction
        xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        np.clip(xy, 0, height, out=xy)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return imw, targets


def convert_images2bmp():
    # cv2.imread() jpg at 230 img/s, *.bmp at 400 img/s
    for path in ['../coco/images/val2014/', '../coco/images/train2014/']:
        folder = os.sep + Path(path).name
        output = path.replace(folder, folder + 'bmp')
        if os.path.exists(output):
            shutil.rmtree(output)  # delete output folder
        os.makedirs(output)  # make new output folder

        for f in tqdm(glob.glob('%s*.jpg' % path)):
            save_name = f.replace('.jpg', '.bmp').replace(folder, folder + 'bmp')
            cv2.imwrite(save_name, cv2.imread(f))

    for label_path in ['../coco/trainvalno5k.txt', '../coco/5k.txt']:
        with open(label_path, 'r') as file:
            lines = file.read()
        lines = lines.replace('2014/', '2014bmp/').replace('.jpg', '.bmp').replace(
            '/Users/glennjocher/PycharmProjects/', '../')
        with open(label_path.replace('5k', '5k_bmp'), 'w') as file:
            file.write(lines)