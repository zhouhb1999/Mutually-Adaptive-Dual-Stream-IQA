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
    def __init__(self, dir, file_name_list, image_size=(256, 256), transform=True):
        self.image_size = image_size
        self.file_name_list = file_name_list
        self.mydir = dir
        self.tranform_toT = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
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

        # change colorspace
        colorspace_choice = np.random.choice([0, 1, 2, 3, 4], 1)[0]
        image_2 = colorspaces(image_2, colorspace_choice)
        image_2 = self.tranform_toT(image_2)

        # scaling transform and random crop
        image = ResizeCrop(image_orig, self.image_size, 3 - div_factor)

        # change colorspace
        colorspace_choice = np.random.choice([0, 1, 2, 3, 4], 1)[0]
        image = colorspaces(image, colorspace_choice)
        image = self.tranform_toT(image)

        # read distortion class, for authentically distorted images it will be 0
        # label=self.file_label_list['score'][idx]
        # label = self.fls.iloc[idx]['labels']
        # label = label[1:-1].split(' ')
        # label = np.array([t.replace(',','') for t in label]).astype(np.float32)

        return image, image_2


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

def loss_fn(q1, q2, z1t, z2t):
    l1 = - tf.cosine_similarity(q1, z1t.detach(), dim=-1).mean()
    l2 = - tf.cosine_similarity(q2, z2t.detach(), dim=-1).mean()

    return (l1 + l2) / 2


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_size=4096, projection_size=256):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    def __init__(self, backbone=None, base_target_ema=0.996, **kwargs):
        super().__init__()
        self.base_ema = base_target_ema

        if backbone is None:
            backbone = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
            # backbone.output_dim = backbone.fc.in_features
            backbone.output_dim = 1000
            backbone.fc = torch.nn.Identity()

        #         encoder = torch.nn.Sequential(*list(backbone.children())[:-1])
        projector = MLPHead(in_dim=backbone.output_dim)

        self.online_encoder = nn.Sequential(
            backbone,
            projector)

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.online_predictor = MLPHead(in_dim=256, hidden_size=1024, projection_size=256)

    @torch.no_grad()
    def update_moving_average(self, global_step, max_steps):

        tau = 1 - ((1 - self.base_ema) * (cos(pi * global_step / max_steps) + 1) / 2)

        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

    def forward(self, x1, x2):

        z1 = self.online_encoder[0](x1)
        z1 = z1['logits']
        z1 = self.online_encoder[1](z1)

        z2 = self.online_encoder[0](x2)
        z2 = z2['logits']
        z2 = self.online_encoder[1](z2)
        q1 = self.online_predictor(z1)
        q2 = self.online_predictor(z2)

        with torch.no_grad():
            z1_t = self.target_encoder[0](x1)
            z1_t=z1_t['logits']
            z1_t = self.target_encoder[1](z1_t)
            z2_t = self.target_encoder[0](x2)
            z2_t=z2_t['logits']
            z2_t = self.target_encoder[1](z2_t)

        loss = loss_fn(q1, q2, z1_t, z2_t)

        return loss
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
    ckpt_dir = "./ckpt/data_swin_384" + 'ava'
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

    model = BYOL().to(device)

    optimizer = LARS(model.named_modules(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    scheduler = LR_Scheduler(
        optimizer, warmup_epochs, warmup_lr * batch_size / 8,

        epochs, lr * batch_size / 8, final_lr * batch_size / 8,
        len(train_loader),
        constant_predictor_lr=True
    )

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
    scaler = torch.cuda.amp.GradScaler()  # 训练前实例化一个GradScaler对象
    for epoch in global_progress:
        print(1)
        train_dataset = image_data(image_path, train_data)
        print(2)
        train_loader = get_train_test_dataloaders(train_dataset, batch_size=batch_size)
        model.train()
        print("{}/{}".format(epoch, epochs))
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')

        for idx, (image, aug_image) in enumerate(local_progress):
            image = image.to(device)
            aug_image = aug_image.to(device)

            model.zero_grad()
            with autocast():
                loss = model.forward(image.to(device, non_blocking=True), aug_image.to(device, non_blocking=True))
            loss_scaler = scaler.scale(loss).item()
            data_dict['loss'] = loss_scaler
            loss_ls.append(loss_scaler)
            scaler.scale(loss).backward()
            if (idx + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, idx + 1, total_steps, loss_scaler
                              ))
            scaler.step(optimizer)  # optimizer.step
            model.update_moving_average(epoch, epochs)
            scaler.update()
            scheduler.step()

            data_dict.update({'lr': scheduler.get_last_lr()})
            local_progress.set_postfix(data_dict)
            logger.update_scalers(data_dict)

        current_loss = data_dict['loss']

        global_progress.set_postfix(data_dict)
        logger.update_scalers(data_dict)

        model_path = os.path.join(ckpt_dir, f"{epoch}.pth")

        if min_loss >= current_loss:
            min_loss = current_loss

            torch.save({
                'epoch': epoch + 1,
                'online_network': model.online_encoder.state_dict(),
                'target_network': model.target_encoder.state_dict()}, model_path)
            print(f'Model saved at: {model_path}')

