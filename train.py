import time
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
# sys.path.append('../global_module/')
# import d2lzh_pytorch as d2l


def evaluate_accuracy(data_iter, net, loss, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, c, y in data_iter:
            test_l_sum, test_num = 0, 0
            X = X.to(device)
            c = c.to(device)
            y = y.to(device)
            net.eval()

            y_hat = net(X,c)
            l = loss(y_hat, y.long())
            acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            test_l_sum += l
            test_num += 1
            net.train() # 改回训练模式
            n += y.shape[0]
    return [acc_sum / n, test_l_sum] # / test_num]

def train(net, train_iter, loss, optimizer, device, Dataset, epochs=30, early_stopping=False,
          early_num=20):
    loss_list = [100]
    early_epoch = 0

    net = net.to(device)
    print("training on ", device)
    start = time.time()
    train_loss_list = []
    train_acc_list = []
    tloss = []
    for epoch in range(epochs):
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        if Dataset == 'UP':
            lr_adjust = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1, last_epoch=-1)
        else:
            lr_adjust = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1, last_epoch=-1)

        for X,c, y in train_iter:
            batch_count, train_l_sum = 0, 0
            X = X.to(device)
            c = c.to(device)
            y = y.to(device)

            y_hat = net(X,c)

            l = loss(y_hat, y.long())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        lr_adjust.step(epoch)

        train_loss_list.append(train_l_sum)
        train_acc_list.append(train_acc_sum / n)

        tloss.append(train_l_sum / batch_count)

        print('epoch %d, train loss %.6f, train acc %.3f, time %.1f sec'
                % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - time_epoch))

        PATH = "./net_DBA.pt"

        if early_stopping and loss_list[-2] < loss_list[-1]:
            if early_epoch == 0:
                torch.save(net.state_dict(), PATH)
            early_epoch += 1
            loss_list[-1] = loss_list[-2]
            if early_epoch == early_num:
                net.load_state_dict(torch.load(PATH))
                break
        else:
            early_epoch = 0

    print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))
