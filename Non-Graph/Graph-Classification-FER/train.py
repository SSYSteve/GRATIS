import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from model.MEFL import GRATIS


def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.dataset == 'FER2013':
        trainset = FER2013(conf.dataset_path, train=True, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 2)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        valset = FER2013(conf.dataset_path, train=False, transform=image_test(crop_size=conf.crop_size), stage = 2)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    return train_loader, val_loader, len(trainset), len(valset)


# Train
def train(conf, net, train_loader, optimizer, epoch):
    losses = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)
    for batch_idx, (inputs,  targets) in enumerate(tqdm(train_loader)):
        outputs = []
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, train_loader_len)
        targets = targets.float()
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        itm_outputs = net(inputs)
        for item in itm_outputs:
            outputs.append(item)
        outputs = torch.stack(outputs)
        device = torch.device('cuda:0')
        outputs = outputs.to(device)
        targets = targets.type(torch.LongTensor)
        Criterion = nn.CrossEntropyLoss()
        loss = Criterion(outputs, targets.to(device))
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))

    return losses.avg


# Val
def val(net, val_loader):
    losses = AverageMeter()
    net.eval()
    y_true = []
    y_pred = []
    f1_score_list = []
    acc_list = []
    test_acc = 0
    valid_class_total = list(0 for i in range(7))
    valid_class_correct = list(0 for i in range(7))

    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        outputs = []
        pred_list = []
        true_list = []
        running_aac = 0
        targets = targets.float()
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            itm_outputs = net(inputs)
            for item in itm_outputs:
                outputs.append(item)
            outputs = torch.stack(outputs)
            device = torch.device('cuda:0')
            outputs = outputs.to(device)
            targets = targets.type(torch.LongTensor)
            Criterion = nn.CrossEntropyLoss()
            loss = Criterion(outputs, targets.to(device))
            losses.update(loss.data.item(), inputs.size(0))
            _, pred = torch.max(outputs,1)
            for i in range(len(pred)):
                valid_class_total[int(targets[i].item())] += 1
                if pred[i].item() == int(targets[i].item()):
                    test_acc += 1
                    running_aac += 1
                    valid_class_correct[int(targets[i].item())] += 1
                y_pred.append(pred[i].item())
                y_true.append(int(targets[i].item()))
                pred_list.append(pred[i].item())
                true_list.append(int(targets[i].item()))
            f1_score_list.append(f1_score(pred_list, true_list, average='macro'))
            acc_list.append((running_aac)/len(pred_list))
    mean_f1_score = f1_score(y_true, y_pred, average='macro')
    mean_acc = (test_acc*100)/len(y_pred)
    return losses.avg, mean_f1_score, f1_score_list, mean_acc, acc_list, valid_class_total, valid_class_correct


def main(conf):
    if conf.dataset == 'FER2013':
        dataset_info = FER2013_infolist

    start_epoch = 0
    # data
    train_loader,val_loader,train_data_num,val_data_num = get_dataloader(conf)
    logging.info("val_data_num: {} ]".format(val_data_num))

    train_losses = []
    valid_losses = []
    acc_list = []

    es = EarlyStopping(patience= 10 )

    net = GRATIS(num_classes=4, backbone=conf.arc)
    logging.info("Resume form | {} ]".format('Stage1_73.06'))
    net = load_state_dict(net, '/content/drive/My Drive/MEFR/sc43/{}.pth'.format(72.63861799944274))

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
    
    optimizer = optim.AdamW(net.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    print('the init learning rate is ', conf.learning_rate)

    #train and val
    for epoch in range(start_epoch, conf.epochs):

        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        train_loss = train(conf,net,train_loader,optimizer,epoch)
        val_loss, val_mean_f1_score, val_f1_score, val_mean_acc, val_acc, valid_class_total, valid_class_correct = val(net, val_loader)

        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        acc_list.append(val_mean_acc)

        best_acc = max(acc_list)
        print('Highest Classification Accuracy: ', best_acc)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        accuracy_history.append(val_mean_acc)

        print('Highest Classification Accuracy of All Time: ', max(accuracy_history))

        if es.step(torch.tensor(val_loss)):
          break  # early stop criterion is met, we can stop now

        # log
        infostr = {'Epoch:  {}   train_loss: {:.5f}  val_loss: {:.5f}  val_mean_f1_score {:.2f},val_mean_acc {:.2f}'
                .format(epoch + 1, train_loss, val_loss,  val_mean_f1_score,  val_mean_acc)}
  
        logging.info(infostr)
        infostr = {'Acc-list:'}
        logging.info(infostr)
        infostr = dataset_info(valid_class_correct, valid_class_total)
        logging.info(infostr)

        # save checkpoints
        if (epoch+1) % 4 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '.pth'))

        checkpoint = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join('/content/drive/My Drive/MEFR/sc43_stage2', '{}'.format(val_mean_acc) + '.pth'))
    
    plt.plot(train_losses,label="Train Loss")
    plt.plot(valid_losses,label="Valid Loss")
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('number of epochs')
    #plt.savefig("SC43_LossPlot/stage2_{}.jpg".format(best_acc))

accuracy_history = []
train_loss_history = []
val_loss_history = []

conf = get_config()

conf.arc = 'resnet50'
conf.batch_size = 64
conf.crop_size = 224
conf.dataset = 'FER2013'
conf.dataset_path = '/content/drive/My Drive/MEFR/data/FER2013'
conf.epochs = 100
conf.evaluate = False
conf.exp_name = 'Resnet50_second_stage'
conf.gpu_ids = '0'
conf.lam = 0.005
conf.learning_rate = 1e-03
conf.metric = 'dots'
conf.neighbor_num = '4'
conf.num_classes = '7'
conf.num_workers = 4
conf.optimizer_eps = 1e-08
conf.resume = ''
conf.seed = 0
conf.weight_decay = 5e-08

set_env(conf)
# generate outdir name
set_outdir(conf)
# Set the logger
set_logger(conf)
main(conf)
