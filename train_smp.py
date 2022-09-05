from multiprocessing.spawn import import_main_path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

import torchvision
from torchvision import transforms as T

import albumentations as A
import pandas as pd
import cv2
from utils import *
from dataset import TianChiDataset
from hrnet.hrnet import HRnet
from loss import *
from tqdm import tqdm
from models import *
from torch import nn
from torch.cuda.amp import autocast
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
import logging
from SegLoss.dice_loss import FocalTversky_loss

logging.basicConfig(filename='log_hrnet_sh_folds.log',
                    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ',
                    level=logging.INFO)

def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()

BATCH_SIZE = 6
IMAGE_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
fp16 = True

trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    # A.RandomCrop(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
    A.OneOf([
        A.RandomContrast(),
        A.RandomGamma(),
        A.RandomBrightness(),
        A.ColorJitter(brightness=0.07, contrast=0.07,
                   saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        ], p=0.3),
])
train_mask = pd.read_csv('/root/DeepLearning/Databases/TianChi/train_mask.csv', sep='\t', names=['name', 'mask'])
train_mask['name'] = train_mask['name'].apply(lambda x: '/root/DeepLearning/Databases/TianChi/train/imgs/' + x)

dataset = TianChiDataset(
    train_mask['name'].values,
    train_mask['mask'].fillna('').values,
    trfm, False, IMAGE_SIZE=IMAGE_SIZE
)

# valid_idx, train_idx = [], []
# for i in range(len(dataset)):
#     if i % 7 == 0:
#         valid_idx.append(i)
# #     else:
#     else:
#         train_idx.append(i)

from sklearn.model_selection import KFold

skf = KFold(n_splits=5)
idx = np.array(range(len(dataset)))


# model = HRnet(num_classes=1)
import segmentation_models_pytorch as smp

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b6",        # 选择解码器, 例如 mobilenet_v2 或 efficientnet-b7
    encoder_weights="imagenet",     # 使用预先训练的权重imagenet进行解码器初始化
    in_channels=3,                  # 模型输入通道（1个用于灰度图像，3个用于RGB等）
    classes=1,                      # 模型输出通道（数据集所分的类别总数）
)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4 * 0.6, weight_decay=1e-3)
# optimizer = torch.optim.ASGD(model.parameters, lr=1e-3 * 0.5, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6, last_epoch=-1)

@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in tqdm(loader):
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        if fp16:
            with autocast():
                output = model(image)
                loss = loss_fn(output, target)
        else:
            output = model(image)
            loss = loss_fn(output, target)
        losses.append(loss.item())
        
    return np.array(losses).mean()


def train(EPOCHES=5, bef_train=None):
    header = r'''
            Train | Valid
    Epoch |  Loss |  Loss | Time, m
    '''
    #          Epoch         metrics            time
    raw_line = '{:6d}' + '\u2502{:7.3f}'*2 + '\u2502{:6.2f}'
    logging.info(header)

    best_loss = 10
    
    if bef_train is not None:
        model.load_state_dict(torch.load(bef_train))
    
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(idx, idx)):
    
        # # #select folder
        # if fold_idx < 1:
        #     continue

        train_ds = D.Subset(dataset, train_idx)
        valid_ds = D.Subset(dataset, valid_idx)

        # define training and validation data loaders
        loader = D.DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        vloader = D.DataLoader(
            valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        
        for epoch in range(1, EPOCHES+1):
            losses = []
            start_time = time.time()
            model.train()
            scaler = torch.cuda.amp.GradScaler()
            for image, target in tqdm(loader):
                
                image, target = image.to(DEVICE), target.float().to(DEVICE)
                optimizer.zero_grad()
                
                if not fp16:
                    output = model(image)
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()
                    
                else:
                    with autocast():
                        output = model(image)
                        loss = loss_fn(output, target)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                losses.append(loss.item())
                # print(loss.item())
                # break
                
            vloss = validation(model, vloader, loss_fn)
            scheduler.step(vloss)
            logging.info(raw_line.format(epoch, np.array(losses).mean(), vloss,
                                  (time.time()-start_time)/60**1))
            losses = []
            
            if vloss < best_loss:
                best_loss = vloss
                torch.save(model.state_dict(), 'model_best_hrnet.pth')
                logging.info(f'Saved Best {vloss} !')

import ttach as tta
def valid(visual_mode=True, use_tta=False):
    trfm = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    ])
    
    as_tensor = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])
    

    subm = []

    model.load_state_dict(torch.load("./model_best_hrnet.pth"))
    model.eval()
    tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean') if use_tta else model

    test_mask = pd.read_csv('/root/DeepLearning/Databases/TianChi/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
    test_mask['name'] = test_mask['name'].apply(lambda x: '/root/DeepLearning/Databases/TianChi/test_a/' + x)

    for idx, name in enumerate(tqdm(test_mask['name'].iloc[:])):
        
        image = cv2.imread(name)
        if visual_mode:
            if idx > 10: break 
            cv2.imwrite(f'./{idx}_raw.jpg', image) 
        image = trfm(image=image)['image']
        image = as_tensor(image)
        with torch.no_grad():
            image = image.to(DEVICE)[None]
            if fp16:
                with autocast():
                    score = tta_model(image)[0][0]
            else:
                score = tta_model(image)[0][0]
            score_sigmoid = score.sigmoid().cpu().numpy()
            score_sigmoid = (score_sigmoid >=0.495).astype(np.uint8)
            score_sigmoid = cv2.resize(score_sigmoid, (512, 512))
            if visual_mode: cv2.imwrite(f'./{idx}_pred.jpg', score_sigmoid * 255)
        # break
        subm.append([name.split('/')[-1], rle_encode(score_sigmoid)])
        
        
    subm = pd.DataFrame(subm)
    subm.to_csv('./effb6_unetpp_fold.csv', index=None, header=None, sep='\t')
    

if __name__ == '__main__':
    # model.load_state_dict(torch.load('./model_best_hrnet.pth'))
    # for _ in range(5):
    #     train(5)
    # train(2)
    valid(visual_mode=False, use_tta=True)