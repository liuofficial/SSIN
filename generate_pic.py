import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import math
from Utils import extract_samll_cubic
import torch.utils.data as Data
from sklearn import preprocessing
from Utils.extract_samll_cubic import select_small_cubic
import cmath

def load_dataset(Dataset, percent):
    if Dataset == 'IN':
        mat_data = sio.loadmat('dataset/Indian_pines_corrected.mat')
        mat_gt = sio.loadmat('dataset/Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        TOTAL_SIZE = 10249
        VALIDATION_SPLIT = percent
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'UP':
        uPavia = sio.loadmat('dataset/PaviaU.mat')
        gt_uPavia = sio.loadmat('dataset/PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776
        VALIDATION_SPLIT = percent
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'SV':
        SV = sio.loadmat('dataset/Salinas_corrected.mat')
        gt_SV = sio.loadmat('dataset/Salinas_gt.mat')
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        TOTAL_SIZE = 54129
        VALIDATION_SPLIT = percent
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'HU':
        HT = sio.loadmat('dataset/HU.mat')
        gt_HT = sio.loadmat('dataset/HU_gt.mat')
        data_hsi = HT['HU']
        gt_hsi = gt_HT['HU_gt']
        TOTAL_SIZE = 30758
        VALIDATION_SPLIT = percent
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'XZ':
        mat_data = sio.loadmat('dataset/xuzhou.mat')
        mat_gt = sio.loadmat('dataset/xuzhou_gt.mat')
        data_hsi = mat_data['xuzhou']
        gt_hsi = mat_gt['xuzhou_gt']
        TOTAL_SIZE = 68877
        VALIDATION_SPLIT = percent
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT

def zhi2polar(heng,zong):
    changdu = []
    hudu = []
    for i, j in zip(heng, zong):
        res = complex(i,j)
        res = cmath.polar(res)

        changdu.append(res[0])
        hudu.append(res[1])
    return changdu, hudu

def save_cmap(img, cmap, fname):
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap)
    plt.savefig(fname, dpi=height)
    plt.close()

def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0

        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes

def sampling2(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if len(indexes) <= 40:
            nb_val = int(len(indexes) / 2)
        else:
            nb_val = proportion
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes

def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)

    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi,
                        ground_truth.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y


def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt):
    gt_all = gt[total_indices] - 1
    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1

    train_index = np.array(train_indices)
    test_index = np.array(test_indices)
    all_index = np.array(total_indices)


    image_x, image_y, _ = whole_data.shape

    n = 2
    train_coor = np.zeros((len(train_index), n))
    test_coor = np.zeros((len(test_index), n))
    all_coor = np.zeros((len(total_indices), n))


    train_coor[:,0] = (train_index // image_y+1)
    train_coor[:,1] = (train_index % image_y +1)

    train_coor[:, 0] = train_coor[:,0] / image_y
    train_coor[:, 1] = train_coor[:,1] / image_x

    test_coor[:,0] = (test_index // image_y+1)
    test_coor[:,1] = (test_index % image_y +1)

    test_coor[:, 0] = test_coor[:, 0] / image_y
    test_coor[:, 1] = test_coor[:, 1] / image_x

    all_coor[:, 0] = (all_index // image_y + 1)
    all_coor[:, 1] = (all_index % image_y + 1)

    all_coor[:, 0] = all_coor[:, 0] / image_y
    all_coor[:, 1] = all_coor[:, 1] / image_x

    all_data = select_small_cubic(TOTAL_SIZE, total_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)

    train_data = select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                    PATCH_LENGTH, padded_data, INPUT_DIMENSION)  # [160, 9, 9, 200]

    test_data = select_small_cubic(TEST_SIZE, test_indices, whole_data,
                                   PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)
    coor_train = train_coor


    x_test = x_test_all
    y_test = y_test
    coor_test = test_coor

    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    x1_tensor_train_c = torch.from_numpy(coor_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, x1_tensor_train_c, y1_tensor_train)

    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    x1_tensor_test_c = torch.from_numpy(coor_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test, x1_tensor_test_c, y1_tensor_test)

    all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION)
    all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    all_tensor_data_c = torch.from_numpy(all_coor).type(torch.FloatTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_c, all_tensor_data_label)


    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=0,
    )

    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,
        num_workers=0,
    )
    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,
        num_workers=0,
    )
    return train_iter, test_iter, all_iter  # , y_test

def generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices, VALIDATION_SPLIT):
    pred_test = []
    for X,c, y in all_iter:
        X = X.to(device)
        c = c.to(device)
        net.eval()
        pred_test.extend(net(X, c).cpu().argmax(axis=1).detach().numpy())
    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            x_label[i] = 16
    gt = gt[:] - 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label)
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)
    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))

    classification_map(y_re, gt_hsi, 300, 'classification_maps/' + Dataset + 'split：' + str(VALIDATION_SPLIT) + '_' + net.name + '.png')
    print('------Get classification maps successful-------')
