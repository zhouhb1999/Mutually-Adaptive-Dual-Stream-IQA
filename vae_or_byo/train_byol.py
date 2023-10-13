
import os
import time
import torch 
import numpy as np
from lars import LARS
from tqdm import tqdm 
from logger import Logger
import torch.optim as optim
from datetime import datetime 
from models.model import BYOL
from dataset import  gpu_transformer
from lr_scheduler import LR_Scheduler
from tensorboardX import SummaryWriter
from dataset import get_train_mem_test_dataloaders
from knn_monitor import knn_monitor as accuracy_monitor


uid = 'byol'
dataset_name = 'stl10'
data_dir = 'dataset'
ckpt_dir = "./ckpt/"+str(datetime.now().strftime('%m%d%H%M%S'))
log_dir = "runs/"+str(datetime.now().strftime('%m%d%H%M%S'))

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    # torch.cuda.set_device(device_id)
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")
    
print(f'Using: {device}')

weight_decay = 1.5e-6
warmup_epochs =  10
warmup_lr = 0
momentum = 0.9
lr =  0.002
final_lr =  0
epochs = 400
stop_at_epoch = 100
batch_size = 64
image_size = (92,92)


train_loader,mem_loader, test_loader = get_train_mem_test_dataloaders(batch_size=batch_size)
train_transform,test_transform = gpu_transformer(image_size)




loss_ls = []
acc_ls = []

model = BYOL().to(device)


optimizer = LARS(model.named_modules(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        
scheduler = LR_Scheduler(
	    optimizer, warmup_epochs, warmup_lr*batch_size/8,

	    epochs, lr*batch_size/8, final_lr*batch_size/8, 
	    len(train_loader),
	    constant_predictor_lr=True 
	    )


min_loss = np.inf 
accuracy = 0

# start training 
logger = Logger(log_dir=log_dir, tensorboard=True, matplotlib=True)
global_progress = tqdm(range(0, epochs), desc=f'Training')
data_dict = {"loss": 100}

for epoch in global_progress:
    model.train()   
    local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
    
    for idx, (image, label) in enumerate(local_progress):
        image = image.to(device)
        aug_image = train_transform(image)
 
        model.zero_grad()
        loss = model.forward(image.to(device, non_blocking=True), aug_image.to(device, non_blocking=True))

        loss_scaler = loss.item()
        data_dict['loss'] = loss_scaler
        loss_ls.append(loss_scaler)
        loss.backward()
        
        optimizer.step()
        model.update_moving_average(epoch, epochs)
        
        scheduler.step()
        
        data_dict.update({'lr': scheduler.get_last_lr()})
        local_progress.set_postfix(data_dict)
        logger.update_scalers(data_dict)
    
    current_loss = data_dict['loss']
    
    global_progress.set_postfix(data_dict)
    logger.update_scalers(data_dict)
    
    model_path = os.path.join(ckpt_dir, f"{uid}_{datetime.now().strftime('%m%d%H%M%S')}.pth")

    if min_loss > current_loss:
        min_loss = current_loss
        
        torch.save({
        'epoch':epoch+1,
        'state_dict': model.state_dict() }, model_path)
        print(f'Model saved at: {model_path}')
