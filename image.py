import h5py
import numpy as np
from PIL import Image
import cv2

def load_data(img_path,train = True):
    #print(img_path)
    while True:
        try:
            if train:
                gt_path = img_path.replace('.jpg','.h5').replace('.bmp','.h5').replace('train_512','train512_gt_density_map')
                #print(gt_path)
            else:
                gt_path = img_path.replace('.jpg', '.h5').replace('.bmp', '.h5').replace('test_512', 'test512_gt_density_map')
                #print(gt_path)
            #gt_path = img_path.replace('.jpg', '.h5').replace('.bmp', '.h5').replace('resize', 'gt_density_map')
            img = Image.open(img_path).convert('RGB')
            gt_file = h5py.File(gt_path)
            target = np.asarray(gt_file['density_map'])
            k = np.asarray(gt_file['kpoint'])
            sigma_map = np.asarray(gt_file['sigma_map'])



            img=img.copy()
            target=target.copy()
            sigma_map = sigma_map.copy()
            k = k.copy()
            break
        except OSError:
            cv2.waitKey(5)

    return img, target, k, sigma_map
