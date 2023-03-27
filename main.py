import numpy as np
import time
import collections
from torch import optim
import torch
from sklearn import metrics, preprocessing
import datetime
from sklearn.decomposition import PCA

import sys
sys.path.append('../global_module/')

import train as train
from generate_pic import aa_and_each_accuracy, sampling, sampling2, load_dataset, generate_png, generate_iter
from Utils import record, extract_samll_cubic
import time
from model.SSIN import SSIN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seeds = [69]*10


ensemble = 1

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')

print('-----Importing Dataset-----')

def main(Dataset, percent):

    data_hsi, gt_hsi,TOTAL_SIZE, TRAIN_SIZE,VALIDATION_SPLIT = load_dataset(Dataset, percent)

    if Dataset in ['XZ']:
        shapeor = data_hsi.shape
        data_hsi = data_hsi.reshape(-1, data_hsi.shape[-1])
        num_components = 20
        print("numcp:", num_components)
        data_hsi = PCA(n_components=num_components).fit_transform(data_hsi)
        shapeor = np.array(shapeor)
        shapeor[-1] = num_components

        data_hsi = data_hsi.reshape(shapeor)

    print(data_hsi.shape)
    image_x, image_y, BAND = data_hsi.shape
    img = data_hsi.reshape(image_x * image_y, BAND)

    img = preprocessing.scale(img)

    pcanum = BAND
    data = img.reshape(image_x*image_y, pcanum)
    gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)

    idx = gt.ravel().tolist()
    num, r = np.unique(idx, return_counts=True)

    sub=0
    CLASSES_NUM = max(gt)
    print('classnum:', CLASSES_NUM)
    print('The class numbers of the HSI data is:', CLASSES_NUM)

    print('-----Importing Setting Parameters-----')
    ITER = 10
    PATCH_LENGTH = 4
    PATCH_SIZE = 2 * PATCH_LENGTH + 1

    lr, num_epochs, batch_size = 0.001, 400, 64
    ikernel, group = 3, 16
    if Dataset == 'UP':
        num_epochs = 300
        group = 16
    if Dataset == 'XZ':
        group = 8
    if Dataset == 'SV':
        group = 32
    if Dataset == 'IN':
        group = 32
    print("Dataset:", Dataset, "epochs:", num_epochs, "group", group)
    loss = torch.nn.CrossEntropyLoss()

    INPUT_DIMENSION = img.shape[-1]

    KAPPA = []
    OA = []
    AA = []
    TRAINING_TIME = []
    TESTING_TIME = []
    ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))


    data_ = data.reshape(image_x, image_y, pcanum)

    whole_data = data_
    padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                             'constant', constant_values=0)

    for index_iter in range(ITER):
        print('iter:', index_iter)
        net = SSIN(BAND, CLASSES_NUM, Dataset, ikernel, group)
        if VALIDATION_SPLIT > 1:
            train_indices, test_indices = sampling2(VALIDATION_SPLIT, gt)
        else:
            train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)

        _, total_indices = sampling(1, gt)

        TRAIN_SIZE = len(train_indices)

        TEST_SIZE = len(test_indices)

        TOTAL_SIZE = len(total_indices)
        print('TRAIN_SIZE:', TRAIN_SIZE, '||', 'TEST_SIZE:', TEST_SIZE, '||', 'TOTAL_SIZE:', TOTAL_SIZE)

        print('-----Selecting Small Pieces from the Original Cube Data-----')

        train_iter, test_iter, all_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices,
                                                                         TOTAL_SIZE, total_indices, whole_data, PATCH_LENGTH,
                                                                         padded_data, INPUT_DIMENSION, batch_size, gt)

        optimizer = optim.SGD(net.parameters(), lr=lr,
                              momentum=0.9,
                              weight_decay=1e-4, nesterov=True)

        np.random.seed(seeds[index_iter])

        tic1 = time.time()
        train.train(net, train_iter, loss, optimizer, device, Dataset, epochs=num_epochs)
        toc1 = time.time()

        pred_test = []
        tic2 = time.time()
        with torch.no_grad():
            for X,c, y in test_iter:
                X = X.to(device)
                c = c.to(device)
                net.eval()
                y_hat = net(X,c)
                pred_test.extend(np.array(net(X,c).cpu().argmax(axis=1)))
        toc2 = time.time()
        collections.Counter(pred_test)
        gt_test = gt[test_indices] - 1


        overall_acc = metrics.accuracy_score(gt_test, pred_test)
        confusion_matrix = metrics.confusion_matrix(gt_test, pred_test)
        each_acc, average_acc = aa_and_each_accuracy(confusion_matrix)
        kappa = metrics.cohen_kappa_score(gt_test, pred_test)

        print("OA", overall_acc)
        print("AA", average_acc)
        print("K", kappa)
        KAPPA.append(kappa)
        OA.append(overall_acc)
        AA.append(average_acc)
        TRAINING_TIME.append(toc1 - tic1)
        TESTING_TIME.append(toc2 - tic2)
        ELEMENT_ACC[index_iter, :] = each_acc

    print("--------" + net.name + " Training Finished-----------")
    record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                         'records/' + net.name + '_' + Dataset + '_s：' + str(PATCH_SIZE) + 'split：' + str(VALIDATION_SPLIT) + 'lr：' + str(
                             lr) + '_k：' + str(ikernel) + '_g：' + str(group) + '.txt')

    generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices, VALIDATION_SPLIT)

if __name__ == '__main__':
    Data = ['UP']
    Per  = [0.995]

    for d in Data:
        for p in Per:
            main(d, p)
            print('='*30)