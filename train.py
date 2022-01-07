from data_read import data_split, Data_Reader
from vnet import VNet
import paddle
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F
import os
import argparse

def dice_loss(score, target):
    target = paddle.cast(target, 'float32')
    smooth = 0.000001
    intersect = paddle.sum(score * target)
    y_sum = paddle.sum(target * target)
    z_sum = paddle.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss_1(score, target):
    score = paddle.flatten(score)
    target = paddle.flatten(target)
    smooth = 0.000001
    intersect = paddle.sum(score * target)
    y_sum = paddle.sum(target)
    z_sum = paddle.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def train(model, iters, optimizer, train_loader, val_loader, log_interval, eval_interval):
    iter = 0
    acc = 0.001
    model.train()
    avg_loss_list = []
    avg_loss_list_1 = []
    while iter < iters :
        iter += 1
        for train_datas in train_loader:
            train_data, train_label = train_datas          
            ligits = model(train_data)
            loss_dice = dice_loss(ligits, train_label)
            loss_cross = F.cross_entropy(ligits, train_label, soft_label=True)
            loss_dice_1 = dice_loss_1(ligits, train_label)
            avg_loss_list.append(loss_cross)
            avg_loss_list_1.append(loss_dice_1)
            loss = loss_dice + loss_cross
            loss.backward()
            optimizer.step()
            model.clear_gradients()
        if iter % log_interval == 0:
            avg_loss = np.array(avg_loss_list).mean()
            avg_loss_1 = np.array(avg_loss_list_1).mean()
            print("[TRAIN] iter={}/{} loss={:.4f}  dice={:.4f}".format(iter, iters, avg_loss, 1-avg_loss_1))
        if iter % eval_interval == 0:
            model.eval()
            avg_loss_list = []
            avg_loss_list_1 = []
            with paddle.no_grad():
                for eval_datas in val_loader:
                    eval_data, eval_label = eval_datas
                    ligits = model(eval_data)
                    loss_dice = dice_loss(ligits, eval_label)
                    loss_cross = F.cross_entropy(ligits, eval_label, soft_label=True)
                    loss_dice_1 = dice_loss_1(ligits, eval_label)
                    avg_loss_list.append(loss_cross)
                    avg_loss_list_1.append(loss_dice_1)
            avg_loss = np.array(avg_loss_list).mean()
            avg_loss_1 = np.array(avg_loss_list_1).mean()
            print("[EVAL] iter={}/{} loss={:.4f} dice={:.4f}".format(iter, iters, avg_loss, 1-avg_loss_1))
            if 1-avg_loss>acc:
                acc = 1-avg_loss
                paddle.save(model.state_dict(),os.path.join("best_model_{:.4f}".format(1-avg_loss_1), 'model.pdparams'))
            model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=4)
    parser.add_argument('--lr',type=float, default=1e-4)
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('Momentum', 'adam'))
    parser.add_argument('--epoch', type=int, default=600)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--root', type=str, default='data/LIDC-IDRI')
    args = parser.parse_args()
    batch_size = args.batchSz
    w = 128
    h = 128
    d = 64
    lr = args.lr
    val_ratio = args.val_ratio
    epoch = args.epoch
    root = args.root
    trainfiles, valfiles = data_split(root, val_ratio)
    _train = Data_Reader(root, trainfiles, w, h, d)
    _val = Data_Reader(root, valfiles, w, h, d)
    model = VNet()
    train_loader = paddle.io.DataLoader(_train, batch_size=batch_size, shuffle=True)
    val_loader = paddle.io.DataLoader(_val, batch_size=batch_size, shuffle=True)
    if args.opt == 'Momentum':
        optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=lr, momentum=0.9,
                                          weight_decay=0.0001)
    if args.opt == 'adam':
        optimizer = paddle.optimizer.Adam(lr,  parameters = model.parameters())   
    train(model, epoch, optimizer, train_loader, val_loader, log_interval = 1, eval_interval=1)

if __name__ == '__main__':
    main()