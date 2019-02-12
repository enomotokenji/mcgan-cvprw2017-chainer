import os
import sys
import numpy as np
import cv2
import random
import glob

import chainer


def read_imlist(root_dir, txt_imlist):
    with open(txt_imlist, 'r') as f:
        ret = [os.path.join(root_dir, path.strip()) for path in f.readlines()]
    return ret


def train_test_dataset(train_class_name, args_train, test_class_name, args_test):
    mod_name = os.path.splitext(os.path.basename(__file__))[0]
    mod_path = os.path.dirname(__file__)
    sys.path.insert(0, mod_path)
    train_class = getattr(__import__(mod_name), train_class_name)
    test_class = getattr(__import__(mod_name), test_class_name)
    train = train_class(**args_train)
    test = test_class(**args_test)

    return train, test


class TestNIRRGB(chainer.dataset.DatasetMixin):
    def __init__(self, dir_nir, dir_rgb, imlist_nir, imlist_rgb):
        super().__init__()
        self.nir = read_imlist(dir_nir, imlist_nir)
        self.rgb = read_imlist(dir_rgb, imlist_rgb)

    def __len__(self):
        return len(self.rgb)

    def get_example(self, i):
        nir = cv2.imread(self.nir[i], 0).astype(np.float32)
        rgb = cv2.imread(self.rgb[i], 1).astype(np.float32)

        nirrgb = np.concatenate((nir[:, :, None], rgb), axis=2)
        nirrgb = nirrgb.transpose(2, 0, 1) / 127.5 - 1.

        return nirrgb,


class TestNIR(chainer.dataset.DatasetMixin):
    def __init__(self, dir_nir, imlist_nir):
        super().__init__()
        self.nir = read_imlist(dir_nir, imlist_nir)

    def __len__(self):
        return len(self.nir)

    def get_example(self, i):
        nir = cv2.imread(self.nir[i], 0).astype(np.float32)
        nir = nir[None, :, :] / 127.5 - 1.

        return nir,


class TestRGB(chainer.dataset.DatasetMixin):
    def __init__(self, dir_rgb, imlist_rgb):
        super().__init__()
        self.rgb = read_imlist(dir_rgb, imlist_rgb)

    def __len__(self):
        return len(self.rgb)

    def get_example(self, i):
        rgb = cv2.imread(self.rgb[i], 1).astype(np.float32)
        rgb = rgb.transpose(2, 0, 1) / 127.5 - 1.

        return rgb,


class BaseTrain(chainer.dataset.DatasetMixin):
    def __init__(self):
        super().__init__()

    def transform(self, x, y):
        c, h, w = x.shape
        if self.augmentation:
            top = random.randint(0, h - self.size - 1)
            left = random.randint(0, w - self.size - 1)
            if random.randint(0, 1):
                x = x[:, :, ::-1]
                y = y[:, :, ::-1]
        else:
            top = (h - self.size) // 2
            left = (w - self.size) // 2
        bottom = top + self.size
        right = left + self.size

        x = x[:, top:bottom, left:right]
        y = y[:, top:bottom, left:right]

        return x, y


class NIRRGB2RGBCLOUD(BaseTrain):
    def __init__(self, dir_nir, dir_rgb, dir_cloud, imlist_nir, imlist_rgb, *args, **kwargs):
        super().__init__()
        self.nir = read_imlist(dir_nir, imlist_nir)
        self.rgb = read_imlist(dir_rgb, imlist_rgb)
        self.cloud = list(glob.glob(os.path.join(dir_cloud, '*.png')))
        self.size = kwargs.pop('size')
        self.augmentation = kwargs.pop('augmentation')

    def __len__(self):
        return len(self.rgb)

    def get_example(self, i):
        nir = cv2.imread(self.nir[i], 0).astype(np.float32)
        rgb = cv2.imread(self.rgb[i], 1).astype(np.float32)
        cloud = cv2.imread(random.choice(self.cloud), -1).astype(np.float32)

        alpha = cloud[:, :, 3] / 255.
        alpha = np.broadcast_to(alpha[:, :, None], alpha.shape + (3,))
        clouded_rgb = (1. - alpha) * rgb + alpha * cloud[:, :, :3]
        clouded_rgb = np.clip(clouded_rgb, 0., 255.)

        nirrgb = np.concatenate((nir[:, :, None], clouded_rgb), axis=2)
        cloud = cloud[:, :, 3]
        rgbcloud = np.concatenate((rgb, cloud[:, :, None]), axis=2)

        nirrgb = nirrgb.transpose(2, 0, 1) / 127.5 - 1.
        rgbcloud = rgbcloud.transpose(2, 0, 1) / 127.5 - 1.

        nirrgb, rgbcloud = self.transform(nirrgb, rgbcloud)

        return nirrgb, rgbcloud


class RGB2RGBCLOUD(BaseTrain):
    def __init__(self, dir_rgb, dir_cloud, imlist_rgb, *args, **kwargs):
        super().__init__()
        self.rgb = read_imlist(dir_rgb, imlist_rgb)
        self.cloud = list(glob.glob(os.path.join(dir_cloud, '*.png')))
        self.size = kwargs.pop('size')
        self.augmentation = kwargs.pop('augmentation')

    def __len__(self):
        return len(self.rgb)

    def get_example(self, i):
        rgb = cv2.imread(self.rgb[i], 1).astype(np.float32)
        cloud = cv2.imread(random.choice(self.cloud), -1).astype(np.float32)

        alpha = cloud[:, :, 3] / 255.
        alpha = np.broadcast_to(alpha[:, :, None], alpha.shape + (3,))
        clouded_rgb = (1. - alpha) * rgb + alpha * cloud[:, :, :3]
        clouded_rgb = np.clip(clouded_rgb, 0., 255.)

        cloud = cloud[:, :, 3]
        rgbcloud = np.concatenate((rgb, cloud[:, :, None]), axis=2)

        rgb = clouded_rgb.transpose(2, 0, 1) / 127.5 - 1.
        rgbcloud = rgbcloud.transpose(2, 0, 1) / 127.5 - 1.

        rgb, rgbcloud = self.transform(rgb, rgbcloud)

        return rgb, rgbcloud


class NIR2RGB(BaseTrain):
    def __init__(self, dir_nir, dir_rgb, imlist_nir, imlist_rgb, *args, **kwargs):
        super().__init__()
        self.nir = read_imlist(dir_nir, imlist_nir)
        self.rgb = read_imlist(dir_rgb, imlist_rgb)
        self.size = kwargs.pop('size')
        self.augmentation = kwargs.pop('augmentation')

    def __len__(self):
        return len(self.rgb)

    def get_example(self, i):
        nir = cv2.imread(self.nir[i], 0).astype(np.float32)
        rgb = cv2.imread(self.rgb[i], 1).astype(np.float32)

        nir = nir[None, :, :] / 127.5 - 1.
        rgb = rgb.transpose(2, 0, 1) / 127.5 - 1.

        nir, rgb = self.transform(nir, rgb)

        return nir, rgb
