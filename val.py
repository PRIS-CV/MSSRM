from __future__ import division

import math
import pickle
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
from torchvision import transforms
import scipy
import dataset
from Networks.MSSRM import MSSRM
from image import *
from torchvision.utils import save_image
import time

warnings.filterwarnings('ignore')
from config import args
import  os
torch.cuda.manual_seed(args.seed)

def main():

    if args.test_dataset == 'ShanghaiA':
        test_file = './npydata/ShanghaiA_test.npy'
    elif args.test_dataset == 'ShanghaiB':
        test_file = './npydata/ShanghaiB_test.npy'
    elif args.test_dataset == 'UCF_QNRF':
        test_file = './npydata/qnrf_test.npy'
    elif args.test_dataset == 'JHU':
        test_file = './npydata/jhu_val.npy'
    elif args.test_dataset == 'NWPU':
        test_file = './npydata/nwpu_val_1024.npy'
    elif args.test_dataset == 'Crowdsr':
        test_file = './npydata/ori512_test.npy'

    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    model = MSSRM(upscale=args.upscale).cuda()
    model = nn.DataParallel(model, device_ids=[0])

    #print(args.pre)
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            # checkpoint = torch.load(args.pre, map_location=lambda storage, loc: storage, pickle_module=pickle)
            checkpoint = torch.load(args.pre)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))


    validate(val_list, model, args)


def target_transform(gt_point, rate):
    point_map = gt_point.cpu().numpy()
    pts = np.array(list(zip(np.nonzero(point_map)[2], np.nonzero(point_map)[1])))
    pt2d = np.zeros((int(rate * point_map.shape[1]) + 1, int(rate * point_map.shape[2]) + 1), dtype=np.float32)

    for i, pt in enumerate(pts):
        pt2d[int(rate * pt[1]), int(rate * pt[0])] = 1.0

    return pt2d


def gt_transform(pt2d, cropsize, rate):
    [x, y, w, h] = cropsize
    pt2d = pt2d[int(y * rate):int(rate * (y + h)), int(x * rate):int(rate * (x + w))]
    density = np.zeros((int(pt2d.shape[0]), int(pt2d.shape[1])), dtype=np.float32)
    pts = np.array(list(zip(np.nonzero(pt2d)[1], np.nonzero(pt2d)[0])))
    orig = np.zeros((int(pt2d.shape[0]), int(pt2d.shape[1])), dtype=np.float32)
    for i, pt in enumerate(pts):
        orig[int(pt[1]), int(pt[0])] = 1.0

    density += scipy.ndimage.filters.gaussian_filter(orig, 4, mode='constant')
    # print(np.sum(density))
    return density

def save_results(input_img, gt_data, density_map, output_dir, fname='results.png'):
    density_map[density_map < 0] = 0
    input_img = input_img[0][0].astype(np.uint8)

    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map[0][0]

    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, 2)
    cv2.imwrite(os.path.join('.', output_dir, fname).replace('.h5', '_1024.jpg').replace('.jpg', '_1024.jpg'),
               density_map)

def validate(Pre_data, model, args):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args.task_id,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]), train=False),
        batch_size=args.batch_size)

    model.eval()

    mae = 0
    mse = 0
    original_mae = 0
    visi = []

    for i, (img, target, kpoint, fname) in enumerate(test_loader):

        img = img.cuda()
        target = target.type(torch.FloatTensor).cuda()

        out1 = model(img, target, None, phase='test')

        count = torch.sum(out1).item()
        gt_count = torch.sum(kpoint).item()
        if i % 50 == 0:
            print(fname[0], 'gt', torch.sum(kpoint).item(), "pred", int(count))
        mae += abs(count - gt_count)
        mse += abs(count - gt_count) * abs(count - gt_count)

    mae = mae / len(test_loader)
    mse = math.sqrt(mse/len(test_loader))

    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}\n'.format(mse=mse))

    return mae, original_mae, visi

if __name__ == '__main__':
    main()
