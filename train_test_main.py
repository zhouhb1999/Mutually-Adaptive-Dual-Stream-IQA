import models
import data_loader
import torch
from scipy import stats
import numpy as np
import torch.optim as optim
from torch.cuda.amp import autocast as autocast
from torch import nn
import argparse
def main(config):
    model = models.Mutual_adaptation_net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    lr=0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader=data_loader.DataLoader(config.dataset,config.img_size, istrain=True)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.002, steps_per_epoch=len(train_loader),
                                                    epochs=config.epochs, div_factor=10)

    criterion = nn.L1Loss()
    # GradScaler对象用来自动做梯度缩放
    scaler = torch.cuda.amp.GradScaler()
    val_srcc_list=[]
    val_plcc_list=[]
    # random crop 10 patches and calculate mean result
    model.train()
    for i, (x_train, y_train, std_train) in enumerate(train_loader):
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        std_train = std_train.to(device)
        optimizer.zero_grad()
        with autocast():
            y_preds, std_preds = model(x_train)
            y_preds = torch.squeeze(y_preds)
            std1 = torch.tensor(std_preds)
            c, w = std1.shape
            mystd = np.ones((c))
            for i in range(c):
                mystd[i] = std1[i, :].std()
            mystd = torch.tensor(mystd)
            mystd = mystd.to(device)
            loss1 = criterion(y_preds, y_train)
            loss2 = criterion(mystd, std_train)
        # k1,k2 is The weight of the quality score loss function and the weight of the standard deviation loss function
        loss= config.k1*loss1 + config.k2*loss2
        loss.requires_grad_(True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    model.eval()
    for i in range(config.train_test_num):
        val_loader = data_loader.DataLoader(config.dataset, config.img_size, istrain=False, isval=True)
        for i, (x_val, y_val, std_val) in enumerate(val_loader):
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            with torch.no_grad():
                y_preds, std_preds = model(x_val)
                y_preds = torch.squeeze(y_preds)

            val_pred_scores = val_pred_scores + y_preds.cpu().float().tolist()
            val_gt_scores = val_gt_scores + y_val.cpu().float().tolist()
        all_srcc, _ = stats.spearmanr(val_pred_scores, val_gt_scores)
        all_plcc, _ = stats.pearsonr(val_pred_scores, val_gt_scores)
        val_srcc_list.append(all_srcc)
        val_plcc_list.append(all_plcc)
    srocc = np.mean(val_srcc_list)
    plcc = np.mean(val_plcc_list)
    print('val predicted quality SROCC: %.2f' % srocc)
    print('val predicted quality PLCC: %.2f' % plcc)

    test_srcc_list=[]
    test_plcc_list=[]
    # random crop 10 patches and calculate mean result
    model.eval()
    for i in range(config.train_test_num):
        test_loader = data_loader.DataLoader(config.dataset, config.img_size, istrain=False, isval=False)
        for i, (x_test, y_test, std_test) in enumerate(test_loader):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            with torch.no_grad():
                y_preds, std_preds = model(x_test)
                y_preds = torch.squeeze(y_preds)

            test_pred_scores = test_pred_scores + y_preds.cpu().float().tolist()
            test_gt_scores = test_gt_scores + y_test.cpu().float().tolist()
        all_srcc, _ = stats.spearmanr(test_pred_scores, test_gt_scores)
        all_plcc, _ = stats.pearsonr(test_pred_scores, test_gt_scores)
        test_srcc_list.append(all_srcc)
        test_plcc_list.append(all_plcc)
    srocc = np.mean(test_srcc_list)
    plcc = np.mean(test_plcc_list)
    print('test predicted quality SROCC: %.2f' % srocc)
    print('test predicted quality PLCC: %.2f' % plcc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='livec', help='Support datasets: livec|koniq-10k|spaq|live|csiq|tid2013|kadid')
    parser.add_argument('--weight_k1', dest='weight_k1', type=float, default=1, help='The weight of the quality score loss function')
    parser.add_argument('--weight_k2', dest='weight_k2', type=int, default=8, help='the weight of the standard deviation loss function')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='Epochs for training')
    parser.add_argument('--_size', dest='img_size', type=int, default=384, help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10, help='Train-test times')

    config = parser.parse_args()
    main(config)