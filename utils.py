import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random
import matplotlib.pyplot as plt
import os
import math
import torch.nn as nn
from skimage import measure
import torch.nn.functional as F
import os
from torch.nn import init
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def seed_pytorch(seed=50):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 and classname.find('SplAtConv2d') == -1:
        init.xavier_normal(m.weight.data)
        
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        
class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x0
                
def random_crop(img, mask, patch_size, pos_prob=None): 
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size)-h),(0, max(w, patch_size)-w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size)-h),(0, max(w, patch_size)-w)), mode='constant')
        h, w = img.shape
        
    while 1:
        h_start = random.randint(0, h - patch_size)
        h_end = h_start + patch_size
        w_start = random.randint(0, w - patch_size)
        w_end = w_start + patch_size

        img_patch = img[h_start:h_end, w_start:w_end]
        mask_patch = mask[h_start:h_end, w_start:w_end]
        
        if pos_prob == None or random.random()> pos_prob:
            break
        elif mask_patch.sum() > 0:
            break
        
    return img_patch, mask_patch

def Normalized(img, img_norm_cfg):
    return (img-img_norm_cfg['mean'])/img_norm_cfg['std']
    
def Denormalization(img, img_norm_cfg):
    return img*img_norm_cfg['std']+img_norm_cfg['mean']

def get_img_norm_cfg(dataset_name, dataset_dir):
    if  dataset_name == 'NUAA-SIRST':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST':
        img_norm_cfg = dict(mean=107.80905151367188, std=33.02274703979492)
    elif dataset_name == 'IRSTD-1K':
        img_norm_cfg = dict(mean=87.4661865234375, std=39.71953201293945)
    elif dataset_name == 'NUDT-SIRST-Sea':
        img_norm_cfg = dict(mean=43.62403869628906, std=18.91838264465332)
    elif dataset_name == 'SIRST4':
        img_norm_cfg = dict(mean=62.10432052612305, std=23.96998405456543)
    elif dataset_name == 'IRDST-real':   
        img_norm_cfg = {'mean': 101.54053497314453, 'std': 56.49856185913086}
    else:
        with open(dataset_dir + '/' + dataset_name +'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            train_list = f.read().splitlines()
        with open(dataset_dir + '/' + dataset_name +'/img_idx/test_' + dataset_name + '.txt', 'r') as f:
            test_list = f.read().splitlines()
        img_list = train_list + test_list
        img_dir = dataset_dir + '/' + dataset_name + '/images/'
        mean_list = []
        std_list = []
        for img_pth in img_list:
            try:
                img = Image.open((img_dir + img_pth).replace('//', '/') + '.png').convert('I')
            except:
                try:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.jpg').convert('I')
                except:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.bmp').convert('I')
            img = np.array(img, dtype=np.float32)
            mean_list.append(img.mean())
            std_list.append(img.std())
        img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
        print(dataset_name + '\t' + str(img_norm_cfg)) 
    return img_norm_cfg

def get_optimizer(net, optimizer_name, scheduler_name, optimizer_settings, scheduler_settings):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'Adagrad':
        optimizer  = torch.optim.Adagrad(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'SGD':
        optimizer  = torch.optim.SGD(net.parameters(), lr=optimizer_settings['lr'])
    
    if scheduler_name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_settings['step'], gamma=scheduler_settings['gamma'])
    elif scheduler_name   == 'CosineAnnealingLR':
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings['epochs'], eta_min=scheduler_settings['min_lr'])
    
    return optimizer, scheduler

def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h//times+1)*times-h),(0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0),(0, (w//times+1)*times-w)), mode='constant')
    return img        


def preprocess(image, x, gap, m):
    """
    将大图进行分块切割，只有当图像尺寸任一方向大于m时才切块
    :param image: 输入的图像，numpy数组
    :param x: 切块大小
    :param gap: 重叠区域长度
    :param m: 尺寸阈值，任一方向上尺寸大于m的图像需要进行切块
    :return: (切块列表, 图像尺寸, 切块位置列表)
    """
    h, w = image.shape[:2]
    
    patches = []
    positions = []

    # 判断是否需要进行切块
    if h <= m and w <= m:
        patches.append(image)
        positions.append((0, 0))
    else:
        # 计算切块的步长（stride）
        stride = x - gap
        
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                # 处理越界的情况
                end_i = min(i + x, h)
                end_j = min(j + x, w)
                
                patch = image[i:end_i, j:end_j]
                patches.append(patch)
                positions.append((i, j))
    
    return patches, (h, w), positions

def postprocess(patches, image_shape, positions, x, gap):
    """
    将分块结果拼接回原图
    :param patches: 切块列表，包含切块的位置信息
    :param image_shape: 原图的尺寸
    :param positions: 切块位置列表
    :param x: 切块大小
    :param gap: 重叠区域长度
    :return: 重建后的图像（Tensor格式）
    """
    h, w = image_shape[:2]
    stride = x - gap
    
    reconstructed_image = torch.zeros((1, 1, h, w), dtype=torch.float32)
    weight_matrix = torch.zeros((1, 1, h, w), dtype=torch.float32)

    for patch, (i, j) in zip(patches, positions):
        end_i = min(i + x, h)
        end_j = min(j + x, w)
        
        
        reconstructed_image[:, :, i:end_i, j:end_j] += patch
        weight_matrix[:, :, i:end_i, j:end_j] += 1
    
    # 处理重叠区域
    reconstructed_image /= weight_matrix
    
    return reconstructed_image


# 示例使用
if __name__ == "__main__":
    # 图像路径列表
    image_paths = ['input_image1.png', 'input_image2.png']
    
    # 切块大小、重叠区域长度和尺寸阈值
    x = 128
    gap = 16
    m = 256
    
    # 创建数据集和数据加载器
    dataset = CustomDataset(image_paths)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    predicted_patches_all = []
    image_shapes = []
    positions_all = []
    
    for image, image_path in tqdm(test_loader):
        image = image[0].numpy()  # 从 DataLoader 获取图像
        
        # 前处理：图像切块
        patches, image_shape, positions = preprocess(image, x, gap, m)
        
        # 收集信息以便后处理
        predicted_patches = []
        for patch in patches:
            img_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(patch, 0), 0)).cuda()
            pred = net.forward(img_tensor)
            pred = pred[0, 0, :patch.shape[0], :patch.shape[1]].cpu().numpy()
            predicted_patches.append(pred)
        
        predicted_patches_all.append(predicted_patches)
        image_shapes.append(image_shape)
        positions_all.append(positions)
        
        # 后处理：图像重建
        reconstructed_image = postprocess(predicted_patches, image_shape, positions, x, gap)
        
        # 保存重建后的图像
        cv2.imwrite(f'reconstructed_{image_path[0]}', reconstructed_image)
