import models
import data_loader
import torch
from scipy import stats
import numpy as np
import torch.optim as optim
model = models.Mutual_adaptation_net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
lr=0.001
srcc_list=[]
plcc_list=[]
# random crop 10 patches and calculate mean result
for i in range(10):
    val_loader = data_loader.DataLoader('live', 384, istrain=False, isval=True)
    for i, (x_train, y_train, std_train) in enumerate(val_loader):
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        std_train = std_train.to(device)
        with torch.no_grad():
            y_preds, std_preds = model(x_train)
            y_preds = torch.squeeze(y_preds)

        test_pred_scores = test_pred_scores + y_preds.cpu().float().tolist()
        test_gt_scores = test_gt_scores + y_train.cpu().float().tolist()
    all_srcc, _ = stats.spearmanr(test_pred_scores, test_gt_scores)
    all_plcc, _ = stats.pearsonr(test_pred_scores, test_gt_scores)
    srcc_list.append(all_srcc)
    plcc_list.append(all_plcc)
srocc = np.mean(srcc_list)
plcc = np.mean(plcc_list)
print('Predicted quality SROCC: %.2f' % srocc)
print('Predicted quality PLCC: %.2f' % plcc)

