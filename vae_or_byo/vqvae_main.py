import os
import numpy as np
import pandas as pd
import os
import sys
import numpy as np
from math import pi, cos
import torch
import torchvision
import torch.nn as nn
from logger import Logger
from torch import allclose
from datetime import datetime
import torch.nn.functional as tf
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.testing import assert_allclose
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
import kornia
from kornia import augmentation as K
import kornia.augmentation.functional as F
import kornia.augmentation.random_generator as rg
from torchvision.transforms import functional as tvF
import scipy.io as scio
import pandas as pd
import scipy.io as scio
import pandas as pd
import torch.nn as nn
import torch
from lr_scheduler import LR_Scheduler

# from torch.optim import lr_scheduler
from lars import LARS
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import ImageCms
from skvideo.utils.mscn import gen_gauss_window
import scipy.ndimage
from PIL import ImageFile
from torch.cuda.amp import autocast as autocast
import random
from trainer import VQVAETrainer

ImageFile.LOAD_TRUNCATED_IMAGES = True


def ResizeCrop(image, sz, div_factor):
    image_size = image.size
    image = transforms.Resize([image_size[1] // div_factor, \
                               image_size[0] // div_factor])(image)

    if image.size[1] < sz[0] or image.size[0] < sz[1]:
        # image size smaller than crop size, zero pad to have same size
        image = transforms.CenterCrop(sz)(image)
    else:
        image = transforms.RandomCrop(sz)(image)

    return image


def compute_MS_transform(image, window, extend_mode='reflect'):
    h, w = image.shape
    mu_image = np.zeros((h, w), dtype=np.float32)
    scipy.ndimage.correlate1d(image, window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, window, 1, mu_image, mode=extend_mode)
    return image - mu_image


def MS_transform(image):
    #   MS Transform
    image = np.array(image).astype(np.float32)
    window = gen_gauss_window(3, 7 / 6)
    image[:, :, 0] = compute_MS_transform(image[:, :, 0], window)
    image[:, :, 0] = (image[:, :, 0] - np.min(image[:, :, 0])) / (np.ptp(image[:, :, 0]) + 1e-3)
    image[:, :, 1] = compute_MS_transform(image[:, :, 1], window)
    image[:, :, 1] = (image[:, :, 1] - np.min(image[:, :, 1])) / (np.ptp(image[:, :, 1]) + 1e-3)
    image[:, :, 2] = compute_MS_transform(image[:, :, 2], window)
    image[:, :, 2] = (image[:, :, 2] - np.min(image[:, :, 2])) / (np.ptp(image[:, :, 2]) + 1e-3)

    image = Image.fromarray((image * 255).astype(np.uint8))
    return image





def colorspaces(im, val):
    if val == 0:
        im = transforms.RandomGrayscale(p=1.0)(im)
    elif val == 1:
        srgb_p = ImageCms.createProfile("sRGB")
        lab_p = ImageCms.createProfile("LAB")

        rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
        im = ImageCms.applyTransform(im, rgb2lab)
    elif val == 2:
        im = im.convert('HSV')
    elif val == 3:
        im = MS_transform(im)
    return im


class image_data(Dataset):
    def __init__(self, dir, file_name_list, image_size=(384, 384), transform=True):
        self.image_size = image_size
        self.file_name_list = file_name_list
        self.mydir = dir
        self.tranform= transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            ])
        self.toT = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])

    def __len__(self):
        self.filelength = len(self.file_name_list)
        return self.filelength

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = self.fls.iloc[idx]['File_names'].rstrip()
        # image_orig = Image.open(img_name)
        img_path = self.file_name_list['file_name'][idx]
        # print(self.mydir+img_path.lstrip('/'))
        image_orig = Image.open(self.mydir + img_path.lstrip('/'))

        if image_orig.mode == 'L':
            image_orig = np.array(image_orig)
            image_orig = np.repeat(image_orig[:, :, None], 3, axis=2)
            image_orig = Image.fromarray(image_orig)
        elif image_orig.mode != 'RGB':
            image_orig = image_orig.convert('RGB')

        # Data augmentations

        # scaling transform and random crop
        div_factor = np.random.choice([1, 2], 1)[0]
        image_2 = ResizeCrop(image_orig, self.image_size, div_factor)
        image_2=self.tranform(image_2)
        # change colorspace
        # colorspace_choice = np.random.choice([0, 1, 2, 3, 4], 1)[0]
        # colorspace_choice = np.random.choice([0, 4], 1)[0]
        # image_1 = colorspaces(image_2, colorspace_choice)

        # image_1 = colorspaces(image_2, 0)
        # gray_img
        # image = transforms.Grayscale(num_output_channels=1)(image_2)
        # image = self.toT(image)
        image_1 = self.toT(image_2)

        return image_1, image_1



import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log2
from typing import Tuple

from helper import HelperModule



def get_train_test_dataloaders(train, batch_size=16, num_workers=4, download=True):
    train_loader = torch.utils.data.DataLoader(
        dataset=train,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers = num_workers,
    )

    return train_loader


from collections import OrderedDict



import copy
from torch import nn
import torchvision.models as models
from transformers import AutoImageProcessor, Swinv2Model
from transformers import AutoImageProcessor, Swinv2ForImageClassification

class InitalTransformation():
    def __init__(self):
        self.transform = T.Compose([
            transforms.RandomCrop((500, 500)),
            torchvision.transforms.RandomCrop(size=image_size),
            T.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])

    def __call__(self, x):
        x = self.transform(x)
        return x


def gpu_transformer(image_size):
    train_transform = nn.Sequential(
        kornia.augmentation.RandomCrop(image_size),
        kornia.augmentation.RandomHorizontalFlip(p=0.5),

    )

    test_transform = nn.Sequential(
        kornia.augmentation.RandomCrop(image_size),
        # kornia.augmentation.RandomHorizontalFlip(p=0.5),

    )

    return train_transform, test_transform


def get_clf_train_test_transform(image_size):
    train_transform = nn.Sequential(

        kornia.augmentation.RandomCrop(image_size),
        kornia.augmentation.RandomHorizontalFlip(p=0.5),

        #               kornia.augmentation.Normalize(CIFAR_MEAN_,CIFAR_STD_),
    )

    test_transform = nn.Sequential(
        kornia.augmentation.RandomCrop(image_size),
        kornia.augmentation.RandomHorizontalFlip(p=0.5),

        # kornia.augmentation.RandomGrayscale(p=0.05),
        # kornia.augmentation.Normalize(CIFAR_MEAN_,CIFAR_STD_)
    )

    return train_transform, test_transform


def read_img_list(path):
    filelist = os.listdir(path)
    list=[]
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            read_img_list(filepath)
        if filename.endswith(".jpg"):
            list.append(filename)
        #print(filename)
    return list

if __name__ == "__main__":
    uid = 'byol'
    dataset_name = 'stl10'
    data_dir = 'dataset'
    # data意思为每次重新做数据增强，防止过拟合
    ckpt_dir = "./ckpt/data_vqvae_384ava_down32" + 'ava'
    log_dir = "runs/data_" + str(datetime.now().strftime('%m%d%H%M%S'))

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    image_size = 384
    # _MEAN_ =  torch.FloatTensor([CIFAR_MEAN])
    # CIFAR_STD_  =  torch.FloatTensor([CIFAR_STD])
    # AVA_path = 'D:/dataset/livec/ChallengeDB_release/Images/'
    AVA_path = 'D:/dataset/ava/AVA_dataset/image/'
    AVA_1024x768 = read_img_list(AVA_path)
    train_data = pd.DataFrame()
    train_data['file_name'] = AVA_1024x768
    train_data = pd.DataFrame(train_data)
    train_data = train_data.reset_index(drop=True)
    image_path = AVA_path
    train_dataset = image_data(image_path, train_data)
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
        device = torch.device("cuda")
        # torch.cuda.set_device(device_id)
    else:
        dtype = torch.FloatTensor
        device = torch.device("cpu")

    print(device)

    weight_decay = 1.5e-6
    warmup_epochs = 10
    warmup_lr = 0
    momentum = 0.9
    lr = 0.002
    final_lr = 0
    epochs = 30
    stop_at_epoch = 30
    batch_size = 64
    knn_monitor = False
    knn_interval = 5
    knn_k = 200

    train_loader = get_train_test_dataloaders(train_dataset, batch_size=batch_size)
    train_transform, test_transform = gpu_transformer(image_size)

    from lr_scheduler import LR_Scheduler

    # from torch.optim import lr_scheduler
    from lars import LARS

    loss_ls = []
    acc_ls = []
    from types import SimpleNamespace

    # model = AutoEncoder().to(device)
    # path = 'C:/Users/Administrator/Desktop/experimemt/ckpt/data_ae_384ava/15.pth'
    # model.load_state_dict(torch.load(path)['model'])
    # optimizer = LARS(model.named_modules(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    #
    # scheduler = LR_Scheduler(
    #     optimizer, warmup_epochs, warmup_lr * batch_size / 8,
    #
    #     epochs, lr * batch_size / 8, final_lr * batch_size / 8,
    #     len(train_loader),
    #     constant_predictor_lr=True
    # )

    from torch.cuda.amp import autocast as autocast
    import random


    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    setup_seed(3407)
    min_loss = np.inf
    accuracy = 0

    # start training
    logger = Logger(log_dir=log_dir, tensorboard=True, matplotlib=True)
    global_progress = tqdm(range(0, epochs), desc=f'Training')
    data_dict = {"loss": 100}
    total_steps = len(train_loader)
    loss_func = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()  # 训练前实例化一个GradScaler对象
    train_dataset = image_data(image_path, train_data)
    train_loader = get_train_test_dataloaders(train_dataset, batch_size=batch_size)
    from tqdm import tqdm
    from torchvision.utils import save_image
    import argparse
    import datetime
    import time
    from pathlib import Path
    from math import sqrt
    from helper import get_device, get_parameter_count
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--task', type=str, default='ava384')
    parser.add_argument('--load-path', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--no-tqdm', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--save-jpg', action='store_true')
    args = parser.parse_args()

    cfg = {
        'display_name': 'ava384',
        'image_shape': (3, 384, 384),

        'in_channels': 3,
        'hidden_channels': 128,
        'res_channels': 64,
        'nb_res_layers': 2,
        'embed_dim': 64,
        'nb_entries': 512,
        'nb_levels': 4,
        'scaling_rates': [4, 2, 2, 2],

        'learning_rate': 1e-4,
        'beta': 0.25,
        'batch_size': 128,
        'mini_batch_size': 128,
        'max_epochs': 66,
    }

    save_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    print(f"> Initialising VQ-VAE-2 model")
    trainer = VQVAETrainer(cfg, args)
    print(f"> Number of parameters: {get_parameter_count(trainer.net)}")

    if args.load_path:
        print(f"> Loading model parameters from checkpoint")
        trainer.load_checkpoint(args.load_path)

    if args.batch_size:
        cfg["batch_size"] = args.batch_size


    if not args.no_save:
        runs_dir = Path(f"runs")
        root_dir = runs_dir / f"{args.task}-{save_id}"
        chk_dir = root_dir / "checkpoints"
        img_dir = root_dir / "images"
        log_dir = root_dir / "logs"

        runs_dir.mkdir(exist_ok=True)
        root_dir.mkdir(exist_ok=True)
        chk_dir.mkdir(exist_ok=True)
        img_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)

    print(f"> Loading {cfg['display_name']} dataset")
    train_dataset = image_data(image_path, train_data)
    train_loader = get_train_test_dataloaders(train_dataset, batch_size=batch_size)

    min_loss = np.inf
    for eid in range(cfg["max_epochs"]):
        print(f"> Epoch {eid + 1}/{cfg['max_epochs']}:")
        epoch_loss, epoch_r_loss, epoch_l_loss = 0.0, 0.0, 0.0
        epoch_start_time = time.time()
        pb = tqdm(train_loader, disable=args.no_tqdm)
        for i, (x, _) in enumerate(pb):
            loss, r_loss, l_loss, _ = trainer.train(x)
            epoch_loss += loss
            epoch_r_loss += r_loss
            epoch_l_loss += l_loss
            pb.set_description(
                f"training_loss: {epoch_loss / (i + 1)} [r_loss: {epoch_r_loss / (i + 1)}, l_loss: {epoch_l_loss / (i + 1)}]")
        print(
            f"> Training loss: {epoch_loss / len(train_loader)} [r_loss: {epoch_r_loss / len(train_loader)}, l_loss: {epoch_l_loss / len(train_loader)}]")

        if min_loss >= epoch_loss:
            best_model_path = os.path.join(ckpt_dir, f"best_{eid+34}.pth")
            torch.save({
                'epoch': eid + 1,
                'model': trainer.net.state_dict()}, best_model_path)
            print(f'Model saved at: {best_model_path}')
        else:
            model_path = os.path.join(ckpt_dir, f"{eid+34}.pth")
            torch.save({
                'epoch': eid + 1,
                'model': trainer.net.state_dict()}, model_path)
            print(f'Model saved at: {model_path}')


        print(f"> Epoch time taken: {time.time() - epoch_start_time:.2f} seconds.")
        print()
