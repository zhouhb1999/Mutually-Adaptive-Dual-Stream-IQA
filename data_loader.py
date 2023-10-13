import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision
from PIL import Image
from torchvision import transforms, datasets
import numpy as np
import os
import scipy.io as scio
import scipy.io
import random

class DataLoader(object):
    """Dataset class for IQA databases"""
    def __init__(self, dataset,img_size,batch_size=1, istrain=True,isval=True,num_workers=4):
        self.batch_size = batch_size
        self.istrain = istrain
        self.isval=isval
        self.num_workers=num_workers
        if dataset == 'live':
            X_train, X_test = create_live_data()
            self.train_dataset, self.test_dataset= create_live_dataset(X_train, X_test, img_size)
        elif dataset == 'livec':
            livec_data = create_livec_data()
            livec_path = 'D:/dataset/livec/ChallengeDB_release/Images/'
            self.train_dataset,self.val_dataset,self.test_dataset = create_clive_dataset(livec_path, livec_data, img_size)
        elif dataset == 'csiq':
            X_train, X_test = create_csiq_data()
            self.train_dataset,self.test_dataset = create_live_dataset(X_train, X_test, img_size)
        elif dataset == 'koniq-10k':
            X_train,X_val,X_test = create_koniq_data()
            koniq_path = 'D:/dataset/koniq/koniq10k_512x384/'
            self.train_dataset,self.val_dataset,self.test_dataset = create_koniq_dataset(koniq_path, X_train,X_val, X_test, img_size)
        elif dataset == 'spaq':
            X_train, X_val, X_test = create_spaqdata()
            self.train_dataset,self.val_dataset,self.test_dataset = create_spaqdataset(X_train, X_val, X_test,img_size)
        elif dataset == 'tid2013':
            X_train, X_test = create_tid2013_data()
            self.train_dataset,self.test_dataset = create_live_dataset(X_train, X_test, img_size)
        elif dataset=='kadid':
            X_train, X_test = create_kadid_data()
            self.train_dataset,self.test_dataset= create_kadid_dataset(X_train, X_test, img_size)
    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.train_dataset, num_workers=self.num_workers,batch_size=self.batch_size, shuffle=True)
        elif self.isval:
            dataloader = torch.utils.data.DataLoader(
                self.val_dataset, num_workers=self.num_workers, batch_size=1)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.test_dataset,num_workers=self.num_workers, batch_size=1)
        return dataloader

def create_csiq_data():
    root='D:/dataset/csiq'
    refpath = os.path.join(root, 'src_imgs')
    refname = getFileName(refpath,'.png')
    txtpath = os.path.join(root, 'csiq_label.txt')
    fh = open(txtpath, 'r')
    imgnames = []
    target = []
    refnames_all = []
    std=[]
    stdpath = os.path.join(root, 'std.txt')
    std_fh = open(stdpath, 'r')
    for line in std_fh:
        line = line.split('\n')
        words = line[0].split()
        std.append(float(words[0]))
    for line in fh:
        line = line.split('\n')
        words = line[0].split()
        imgnames.append((words[0]))
        target.append(words[1])
        ref_temp = words[0].split(".")
        refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])

    labels = np.array(target).astype(np.float32)
    refnames_all = np.array(refnames_all)

    train = []
    test=[]
    # index =list(range(0, 29))
    patch_num=1
    import random
    sel_num = list(range(0, 30))
    random.shuffle(sel_num)
    train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
    test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

    for i, item in enumerate(train_index):
        train_sel = (refname[train_index[i]] == refnames_all)
        train_sel = np.where(train_sel == True)
        train_sel = train_sel[0].tolist()
        for j, item in enumerate(train_sel):
            for aug in range(patch_num):
                train.append((os.path.join(root, 'dst_imgs', imgnames[item]), labels[item],std[item]))
    for i, item in enumerate(test_index):
        train_sel = (refname[test_index[i]] == refnames_all)
        train_sel = np.where(train_sel == True)
        train_sel = train_sel[0].tolist()
        for j, item in enumerate(train_sel):
            for aug in range(patch_num):
                test.append((os.path.join(root, 'dst_imgs', imgnames[item]), labels[item],std[item]))
    return train,test
def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename
def create_tid2013_data():
    root = 'D:/dataset/tid2013'
    refpath = os.path.join(root, 'reference_images')
    refname = getTIDFileName(refpath, '.bmp.BMP')
    txtpath = os.path.join(root, 'mos_with_names.txt')
    fh = open(txtpath, 'r')
    imgnames = []
    target = []
    refnames_all = []
    stdpath=os.path.join(root, 'mos_std.txt')
    fh_std=open(stdpath, 'r')
    std_label=[]
    for line in fh_std:
        line = line.split('\n')
        words = line[0].split()
        std_label.append(float(words[0]))
    for line in fh:
        line = line.split('\n')
        words = line[0].split()
        imgnames.append((words[1]))
        target.append(words[0])
        ref_temp = words[1].split("_")
        refnames_all.append(ref_temp[0][1:])
    labels = np.array(target).astype(np.float32)
    refnames_all = np.array(refnames_all)
    sel_num = list(range(0, 25))
    random.shuffle(sel_num)
    train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
    test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
    train = []
    test = []
    patch_num=1
    for i, item in enumerate(train_index):
        train_sel = (refname[train_index[i]] == refnames_all)
        train_sel = np.where(train_sel == True)
        train_sel = train_sel[0].tolist()
        for j, item in enumerate(train_sel):
            for aug in range(patch_num):
                train.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item],std_label[item]))
    for i, item in enumerate(test_index):
        train_sel = (refname[test_index[i]] == refnames_all)
        train_sel = np.where(train_sel == True)
        train_sel = train_sel[0].tolist()
        for j, item in enumerate(train_sel):
            for aug in range(patch_num):
                test.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item],std_label[item]))
    return train,test
def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
def getDistortionTypeFileName(path, num):
     filename = []
     index = 1
     for i in range(0, num):
          name = '%s%s%s' % ('img', str(index), '.bmp')
          filename.append(os.path.join(path, name))
          index = index + 1
     return filename

def create_live_data():
    root='D:/dataset/live-iqa'
    refpath = os.path.join(root, 'refimgs')

    refname = getFileName(refpath, '.bmp')

    jp2kroot = os.path.join(root, 'jp2k')
    jp2kname = getDistortionTypeFileName(jp2kroot, 227)

    jpegroot = os.path.join(root, 'jpeg')
    jpegname = getDistortionTypeFileName(jpegroot, 233)

    wnroot = os.path.join(root, 'wn')
    wnname = getDistortionTypeFileName(wnroot, 174)

    gblurroot = os.path.join(root, 'gblur')
    gblurname = getDistortionTypeFileName(gblurroot, 174)

    fastfadingroot = os.path.join(root, 'fastfading')
    fastfadingname = getDistortionTypeFileName(fastfadingroot, 174)

    imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname
    dmos = scipy.io.loadmat(os.path.join(root, 'dmos.mat'))
    labels = dmos['dmos'].astype(np.float32)
    orgs = dmos['orgs']
    refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))
    refnames_all = refnames_all['refnames_all']
    train = []
    test=[]
    # index =list(range(0, 29))
    patch_num=1
    import random
    sel_num = list(range(0, 29))
    random.shuffle(sel_num)
    train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
    test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
    for i in range(0, len(train_index)):
        train_sel = (refname[train_index[i]] == refnames_all)
        train_sel = train_sel * ~orgs.astype(np.bool_)
        train_sel = np.where(train_sel == True)
        train_sel = train_sel[1].tolist()
        for j, item in enumerate(train_sel):
            for aug in range(patch_num):
                train.append((imgpath[item], labels[0][item]))
    for i in range(0, len(test_index)):
        train_sel = (refname[test_index[i]] == refnames_all)
        train_sel = train_sel * ~orgs.astype(np.bool_)
        train_sel = np.where(train_sel == True)
        train_sel = train_sel[1].tolist()
        for j, item in enumerate(train_sel):
            for aug in range(patch_num):
                test.append((imgpath[item], labels[0][item]))
    return train,test
class live_dataset(torch.utils.data.Dataset):
    def __init__(self, data,transform=None):
        self.data = data
        self.transform = transform
    # dataset length
    def __len__(self):
        self.filelength = len(self.data)
        return self.filelength

    # load an one of images
    def __getitem__(self, idx):
        img_path = self.data[idx][0]
        img = Image.open(img_path.lstrip('/'))
        img = img.convert('RGB')
        img_transformed = self.transform(img)
        label = self.data[idx][1]
        std = self.data[idx][2]
        return img_transformed, label,std
def create_live_dataset(train_data,test_data,image_size):

    data_transform = {
        "train": transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.Resize((512, 384)),
            torchvision.transforms.RandomCrop(size=image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ]),  # 来自官网参数
        "val": transforms.Compose([
            # torchvision.transforms.Resize((512, 384)),
            # torchvision.transforms.RandomCrop(size=image_size),
            torchvision.transforms.CenterCrop(size=image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])}
    train_dataset= live_dataset(train_data, transform=data_transform["train"])
    test_dataset = live_dataset(test_data, transform=data_transform["val"])
    return train_dataset,test_dataset
def create_kadid_data():
    root='D:/dataset/kadid10k/'
    data = pd.read_csv('D:/dataset/kadid10k/dmos.csv')
    sel_num = []
    for i in range(81):
        if i <= 8:
            sel_num.append('I' + str(0) + str(i + 1) + '.png')
        else:
            sel_num.append('I' + str(i + 1) + '.png')
    random.shuffle(sel_num)
    train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
    test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
    train=data.loc[data['ref_img'].isin(train_index)]
    test=data.loc[data['ref_img'].isin(test_index)]
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train,test
class kadid_dataset(torch.utils.data.Dataset):
    def __init__(self, data,transform=None):
        self.data = data
        self.transform = transform


    # dataset length
    def __len__(self):
        self.filelength = len(self.data)
        return self.filelength

    # load an one of images
    def __getitem__(self, idx):
        img_path = self.data['dist_img'][idx]
        img = Image.open("D:/dataset/kadid10k/images/"+img_path.lstrip('/'))
        img = img.convert('RGB')
        img_transformed = self.transform(img)
        label = self.data['dmos'][idx]
        std=self.data['var'][idx]
        return img_transformed, label,std
def create_kadid_dataset(train_data,test_data,image_size):

    data_transform = {
        "train": transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.Resize((512, 384)),
            torchvision.transforms.RandomCrop(size=image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ]),  # 来自官网参数
        "val": transforms.Compose([
            # torchvision.transforms.Resize((512, 384)),
            torchvision.transforms.CenterCrop(size=image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])}
    train_dataset= kadid_dataset(train_data, transform=data_transform["train"])
    test_dataset = kadid_dataset(test_data, transform=data_transform["val"])
    return train_dataset,test_dataset
class kadid_dataset(torch.utils.data.Dataset):
    def __init__(self, data,transform=None):
        self.data = data
        self.transform = transform


    # dataset length
    def __len__(self):
        self.filelength = len(self.data)
        return self.filelength

    # load an one of images
    def __getitem__(self, idx):
        img_path = self.data['dist_img'][idx]

        # This is the location of your kadid database
        kadid_img_path='D:/dataset/kadid10k/images/'

        img = Image.open(kadid_img_path+img_path.lstrip('/'))
        img = img.convert('RGB')
        img_transformed = self.transform(img)
        label = self.data['dmos'][idx]
        std=self.data['var'][idx]
        return img_transformed, label,std

def create_livec_data():
    # This is the location of your livec database
    labelFile = 'D:/dataset/livec/ChallengeDB_release/Data/AllMOS_release.mat'
    label = scio.loadmat(labelFile)
    label = label['AllMOS_release']

    dataFile = 'D:/dataset/livec/ChallengeDB_release/Data/AllImages_release.mat'
    data = scio.loadmat(dataFile)
    data = data['AllImages_release']

    stdfile = 'D:/dataset/livec/ChallengeDB_release/Data/AllStdDev_release.mat'
    std = scio.loadmat(stdfile)
    std = std['AllStdDev_release']

    # 前七个是trainingimages文件夹中的，这里我们将这七个当做测试，在训练阶段不看

    test_filename = data[0:7]
    train_filename = data[7:1170]

    test_label = label[0][0:7]
    train_label = label[0][7:1170]

    test_stdlabel = std[0][0:7]
    train_stdlabel = std[0][7:1170]


    train_data = pd.DataFrame()
    train_data['file_name'] = 0
    for i in range(len(train_label)):
        train_data = train_data.append({'file_name': train_filename[i][0][0]}, ignore_index=True)

    train_data['score'] = train_label
    train_data['std'] = train_stdlabel

    data=pd.DataFrame(train_data)
    return data
def create_clive_dataset(data_dir,train_data,image_size):
    # 划分数据集时需要随机数种子一样
    X_train, X_test= train_test_split(train_data, test_size=0.3, random_state=3407)
    X_train = pd.DataFrame(X_train)
    X_train = X_train.reset_index(drop=True)
    X_test = pd.DataFrame(X_test)
    X_test = X_test.reset_index(drop=True)


    X_val, X_test= train_test_split(X_test, test_size=2 / 3, random_state=3407)

    X_val = pd.DataFrame(X_val)
    X_val = X_val.reset_index(drop=True)
    X_test = pd.DataFrame(X_test)
    X_test = X_test.reset_index(drop=True)
    data_transform = {
        "train": transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ]),  # 来自官网参数
        "val": transforms.Compose([
            torchvision.transforms.CenterCrop(size=image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])}
    train_dataset= livec_traindataset(data_dir,X_train,image_size, transform=data_transform["train"])
    val_dataset = livec_traindataset(data_dir, X_val, image_size, transform=data_transform["val"])
    test_dataset = livec_traindataset(data_dir, X_test, image_size, transform=data_transform["val"])
    return train_dataset,val_dataset,test_dataset
class livec_traindataset(torch.utils.data.Dataset):

    def __init__(self, dir, data, image_size,transform):
        self.data = data
        self.mydir = dir
        self.image_size = image_size
        self.transform=transform
    # dataset length
    def __len__(self):
        self.filelength = len(self.data)
        return self.filelength

    # load an one of images
    def __getitem__(self, idx):
        img_path = self.data['file_name'][idx]
        img = Image.open(self.mydir + img_path.lstrip('/'))
        img = img.convert('RGB')

        # width, height = img.size
        # if height>width:
        #     height=width

        img_transformed = self.transform(img)
        label = self.data['score'][idx]
        std = self.data['std'][idx]
        # std=self.data['SD'][idx]

        return img_transformed, label, std

def create_koniq_data():
    # This is the labeled location of your KonIQ dataset
    koniq_path = 'D:/dataset/koniq/koniq10k_scores_and_distributions.csv'
    head_row = pd.read_csv(koniq_path, nrows=0)
    head_row_list = list(head_row)
    csv_result = pd.read_csv(koniq_path, usecols=head_row_list)
    # 划分数据集时需要随机数种子一样
    X_train, X_test = train_test_split(csv_result, test_size=0.3, random_state=0)

    X_train = pd.DataFrame(X_train)
    X_train = X_train.reset_index(drop=True)
    X_test = pd.DataFrame(X_test)
    X_test = X_test.reset_index(drop=True)

    X_val, X_test = train_test_split(X_test,
                                     test_size=2 / 3, random_state=3407)

    X_val = pd.DataFrame(X_val)
    X_val = X_val.reset_index(drop=True)
    X_test = pd.DataFrame(X_test)
    X_test = X_test.reset_index(drop=True)
    return X_train,X_val,X_test

def create_koniq_dataset(data_dir,train_data,val_data,test_data,image_size):
    # data_dir is your location of your KonIQ dataset
    data_transform = {
        "train": transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.Resize((512, 384)),
            torchvision.transforms.RandomCrop(size=image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ]),  # 来自官网参数
        "val": transforms.Compose([
            torchvision.transforms.CenterCrop(size=image_size),
            # torchvision.transforms.RandomCrop(size=image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])}
    train_dataset= koniq_dataset(data_dir,train_data, transform=data_transform["train"])
    val_dataset = koniq_dataset(data_dir, val_data, transform=data_transform["val"])
    test_dataset = koniq_dataset(data_dir, test_data, transform=data_transform["val"])

    return train_dataset,val_dataset,test_dataset

class koniq_dataset(torch.utils.data.Dataset):
    def __init__(self, dir, data,transform=None,choice=False):
        self.data = data
        self.transform = transform
        self.mydir = dir

    # dataset length
    def __len__(self):
        self.filelength = len(self.data)
        return self.filelength

    # load an one of images
    def __getitem__(self, idx):
        img_path = self.data['image_name'][idx]
        img = Image.open(self.mydir + img_path.lstrip('/'))
        img = img.convert('RGB')

        img_transformed = self.transform(img)
        label = self.data['MOS_zscore'][idx]
        # MOS = self.data['MOS'][idx]
        std = self.data['SD'][idx]
        # c1=self.data['c1'][idx]
        # c2 = self.data['c2'][idx]
        # c3 = self.data['c3'][idx]
        # c4 = self.data['c4'][idx]
        # c5 = self.data['c5'][idx]
        # labellist=[c1,c2,c3,c4,c5]
        # labellist = torch.as_tensor(labellist)
        return img_transformed, label, std

def create_spaqdata():
    # spaq_path is the label storage location in your SPAQ database
    spaq_path = 'D:/dataset/spaq/Annotations/MOS and Image attribute scores.xlsx'
    data = pd.read_excel(spaq_path, sheet_name=0)

    # 划分数据集时需要随机数种子一样
    X_train, X_test = train_test_split(data, test_size=0.3, random_state=3407)

    X_train = pd.DataFrame(X_train)
    X_train = X_train.reset_index(drop=True)
    X_test = pd.DataFrame(X_test)
    X_test = X_test.reset_index(drop=True)

    X_val, X_test = train_test_split(X_test,
                                     test_size=2 / 3, random_state=3407)

    X_val = pd.DataFrame(X_val)
    X_val = X_val.reset_index(drop=True)
    X_test = pd.DataFrame(X_test)
    X_test = X_test.reset_index(drop=True)
    return X_train, X_val, X_test

class spaqdataset(torch.utils.data.Dataset):

    def __init__(self, dir, data, image_size,transforms,choice=False):
        self.data = data
        self.mydir = dir
        self.image_size = image_size
        self.transform=transforms
        self.choice=choice

    # dataset length
    def __len__(self):
        self.filelength = len(self.data)
        return self.filelength

    # load an one of images
    def __getitem__(self, idx):
        img_path = self.data['Image name'][idx]
        img = Image.open(self.mydir + img_path.lstrip('/'))
        img = img.convert('RGB')
        img_transformed = self.transform(img)
        label = self.data['MOS'][idx]

        Brightness = self.data['Brightness'][idx]
        Colorfulness = self.data['Colorfulness'][idx]
        Contrast = self.data['Contrast'][idx]
        Noisiness = self.data['Noisiness'][idx]
        Sharpness = self.data['Contrast'][idx]

        labellist = [Brightness, Colorfulness, Contrast, Noisiness, Sharpness]
        labellist = torch.Tensor(labellist)


        return img_transformed, label, labellist
def create_spaqdataset(X_train, X_val, X_test,img_size):
    # image_size = 384
    image_size=img_size
    data_transform = {
        "train": transforms.Compose([
            torchvision.transforms.Resize(size=image_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ]),  # 来自官网参数
        "val": transforms.Compose([
            torchvision.transforms.Resize(size=image_size),
            torchvision.transforms.CenterCrop(size=image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ]),
    }
    # image_path is the location of your SPAQ dataset
    image_path = 'C:/Users/Administrator/Desktop/实验/Anti-aliasing/'

    train_dataset = spaqdataset(image_path, X_train, image_size,data_transform["train"])
    val_dataset = spaqdataset(image_path, X_val, image_size,data_transform["val"])
    test_dataset = spaqdataset(image_path, X_test, image_size,data_transform["val"])
    return train_dataset, val_dataset, test_dataset
