import os
import numpy as np

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')


'''please set your dataset path'''

try:
    sr_train_path='/data/xiejiahao/Crowd_SR/train/train_512/'
    sr_test_path='/data/xiejiahao/Crowd_SR/test/test_512/'

    train_list = []
    for filename in os.listdir(sr_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(sr_train_path+filename)
    train_list.sort()
    np.save('./npydata/ori512_train.npy', train_list)


    test_list = []
    for filename in os.listdir(sr_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(sr_test_path+filename)
    test_list.sort()
    np.save('./npydata/ori512_test.npy', test_list)
    print("Generate ori512 image list successfully")
except:
    print("The sr dataset path is wrong. Please check your path.")
