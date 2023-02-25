from __future__ import division
import warnings


from Networks.MSSRM import MSSRM
from utils import save_checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import dataset
import math
from image import *

warnings.filterwarnings('ignore')
from config import args
import  os
import scipy.misc
import imageio
import time
import random
import scipy.ndimage
import cv2
torch.cuda.manual_seed(args.seed)

print(args)

def main():
    train_file = './npydata/ori512_train.npy'
    val_file = './npydata/ori512_test.npy'

    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(val_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()



    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    model = MSSRM(upscale=args.upscale)
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    criterion =  nn.MSELoss(size_average=False).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=10, verbose=True)
    print(args.pre)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            args.best_pred =  checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
            args.best_pred = 10000

    torch.set_num_threads(args.workers)

    print(args.best_pred)

    if not os.path.exists(args.task_id):
        os.makedirs(args.task_id)

    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()
        adjust_learning_rate(optimizer, epoch)


        train(train_list, model, criterion, optimizer, epoch, args,scheduler )
        end_train = time.time()
        print("train time ", end_train-start)
        # if epoch<30:
        #     continue
        prec1, visi = validate(val_list, model, args)

        is_best = prec1 < args.best_pred
        args.best_pred = min(prec1, args.best_pred)

        print(' * best MAE {mae:.3f} '
              .format(mae=args.best_pred))


        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': args.best_pred,
            'optimizer': optimizer.state_dict(),
        }, visi, is_best, args.task_id)
        end_val = time.time()
        print("val time",end_val - end_train)


        # if is_best==True:
        #     prec1, visi = validate(test_list, model, args)



def crop(d, g):
    g_h, g_w = g.size()[2:4]
    d_h, d_w = d.size()[2:4]

    d1 = d[:, :, abs(int(math.floor((d_h - g_h) / 2.0))):abs(int(math.floor((d_h - g_h) / 2.0))) + g_h,
         abs(int(math.floor((d_w - g_w) / 2.0))):abs(int(math.floor((d_w - g_w) / 2.0))) + g_w]
    return d1


def choose_crop(output, target):
    if (output.size()[2] > target.size()[2]) | (output.size()[3] > target.size()[3]):
        output = crop(output, target)
    if (output.size()[2] > target.size()[2]) | (output.size()[3] > target.size()[3]):
        output = crop(output, target)
    if (output.size()[2] < target.size()[2]) | (output.size()[3] < target.size()[3]):
        target = crop(target, output)
    if (output.size()[2] < target.size()[2]) | (output.size()[3] < target.size()[3]):
        target = crop(target, output)
    return output, target



def gt_transform(pt2d, rate):
    # print(pt2d.shape,rate)
    pt2d = pt2d.data.numpy()

    density = np.zeros((int(rate * pt2d.shape[0]) + 1, int(rate * pt2d.shape[1]) + 1))
    pts = np.array(list(zip(np.nonzero(pt2d)[1], np.nonzero(pt2d)[0])))

    # print(pts.shape,np.nonzero(pt2d)[1],np.nonzero(pt2d)[0])
    orig = np.zeros((int(rate * pt2d.shape[0]) + 1, int(rate * pt2d.shape[1]) + 1))

    for i, pt in enumerate(pts):
        #    orig = np.zeros((int(rate*pt2d.shape[0])+1,int(rate*pt2d.shape[1])+1),dtype=np.float32)
        orig[int(rate * pt[1]), int(rate * pt[0])] = 1.0
    #    print(pt)

    density += scipy.ndimage.filters.gaussian_filter(orig, 8)

    # density_map = density
    # density_map = density_map / np.max(density_map) * 255
    # density_map = density_map.astype(np.uint8)
    # density_map = cv2.applyColorMap(density_map, 2)
    # cv2.imwrite('./temp/1.jpg', density_map)

    # print(np.sum(density))
    # print(pt2d.sum(),pts.shape, orig.sum(),density.sum())
    return density

def train(Pre_data, model, criterion, optimizer, epoch, args, scheduler):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()


    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args.task_id,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            seen=model.module.seen,
                            batch_size=args.batch_size,
                            num_workers=args.workers, phase=args.upscale),
        batch_size=args.batch_size, drop_last=False)
    #print(train_loader)
    args.lr = optimizer.param_groups[0]['lr']
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()
    loss_ave = 0.0

    for i, (img, target, kpoint, fname, img_sr) in enumerate(train_loader):
        #print('img3:', img.size(), 'target:', target.size())
        data_time.update(time.time() - end)
        img = img.cuda()
        img_sr = img_sr.cuda()

        target = target.type(torch.FloatTensor).cuda()

        d5, out_sr = model(img, target, img_sr, phase='train')


        target = target.unsqueeze(1)

        loss = criterion(d5, target) + 1e-08 * criterion(out_sr, img_sr)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('4_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
        loss_ave  += loss.item()
    loss_ave = loss_ave*1.0/len(train_loader)

    print(loss_ave, args.lr)
    #scheduler.step(loss_ave)

def validate(Pre_data, model, args):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args.task_id,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]), train=False),
        batch_size=1)

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []

    for i, (img, target, kpoint, fname) in enumerate(test_loader):
        img = img.cuda()
        target = target.type(torch.FloatTensor).cuda()
        d5 = model(img, target, None, phase='test')

        count = torch.sum(d5).item()
        mae += abs(torch.sum(kpoint).item() - count)
        mse += abs(torch.sum(kpoint).item() - count) * abs(torch.sum(kpoint).item() - count)
        if i % 50 == 0:
            print(fname[0], 'gt', torch.sum(kpoint).item(), "pred", int(count))
            visi.append(
                [img.data.cpu().numpy(), d5.data.cpu().numpy(), target.unsqueeze(0).data.cpu().numpy(),
                 fname])

    mae = mae * 1.0/ len(test_loader)
    mse = math.sqrt(mse/len(test_loader))

    print(' \n* MAE {mae:.3f}\n'.format(mae=mae),'* MSE {mse:.3f}'.format(mse=mse))

    return mae, visi

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    # if epoch > 100:
    #     args.lr = 1e-5
    # if epoch > 300:
    #     args.lr = 1e-5


    # for i in range(len(args.steps)):
    #
    #     scale = args.scales[i] if i < len(args.scales) else 1
    #
    #     if epoch >= args.steps[i]:
    #         args.lr = args.lr * scale
    #         if epoch == args.steps[i]:
    #             break
    #     else:
    #         break
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = args.lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
