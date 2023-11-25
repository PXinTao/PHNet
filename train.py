import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
import numpy as np
from argparse import ArgumentParser
import transforms
import mstr
from data import BSDS_500, NYUD, PASCAL_Context, PASCAL_VOC12
import model
import cv2
import time
import os
from datetime import datetime
import deep_xy


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-r", "--resume", action="store_true", default=False)
    # parser.add_argument("-g", "--gpu", type=int, default=0)
    # parser.add_argument("-t", "--task", type=str, default='stairfuse-original')
    args = parser.parse_args()
    # load configures
    file_id = open('./cfgs.yaml')
    cfgs = yaml.load(file_id, Loader=yaml.SafeLoader)
    file_id.close()

    trans = transforms.Compose([
        # transforms.RandomScale((0.7, 1.3)),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomResizedCrop(320, scale=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # dataset = PASCAL_Context(root=cfgs['dataset'], flag='train', transform=trans)
    # dataset = BSDS_500(root=cfgs['dataset'], VOC=True, transform=trans)
    # dataset = PASCAL_VOC12(root=cfgs['dataset'], transform=trans)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfgs['batch_size'], shuffle=True, num_workers=4)

    dataset = BSDS_500(root=cfgs['dataset'], VOC=True, transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfgs['batch_size'], shuffle=True, num_workers=1)

    # model
    # net = model.MSFuse(cfgs).train()
    net = deep_xy.VisualNet().train()
    # net = modelRes.DRNet().train()
    # loss
    criterion = model.Cross_Entropy()
    # optimal
    if cfgs['method'] == 'Adam':
        optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': criterion.parameters()}],lr=cfgs['lr'], weight_decay=cfgs['weight_decay'])
    elif cfgs['method'] == 'SGD':
        optimizer = torch.optim.SGD([{'params': net.parameters()}, {'params': criterion.parameters()}], lr=cfgs['lr'], momentum=cfgs['momentum'], weight_decay=cfgs['weight_decay'])
    elif cfgs['method'] == 'AdamW':
        optimizer = torch.optim.AdamW([{'params': net.parameters()}, {'params': criterion.parameters()}], lr=cfgs['lr'], weight_decay=cfgs['weight_decay'])
    # # multi_GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     net = torch.nn.DataParallel(net)
    net.to(device)
    criterion.to(device)
    
    start_epoch = 0
    if args.resume:
        state = torch.load('checkpoint.pth')
        start_epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        net.load_state_dict(state['param'])
        # scheduler.load_state_dict(state['scheduler'])
        del state
    net.train()

    # train
    for epoch in range(start_epoch, cfgs['max_iter']):  # loop over the dataset multiple times
        # model.learning_rate_decay(optimizer, epoch, decay_rate=cfgs['decay_rate'], decay_steps=cfgs['decay_steps'])
        if epoch == 9:
            optimizer = torch.optim.AdamW([{'params': net.parameters()}, {'params': criterion.parameters()}],
                                          lr=cfgs['lr']*0.1, weight_decay=cfgs['weight_decay'])
        # if epoch == 5:
        #     optimizer = torch.optim.AdamW([{'params': net.parameters()}, {'params': criterion.parameters()}],
        #                                   lr=cfgs['lr']*0.01, weight_decay=cfgs['weight_decay'])
        running_loss = 0.0
        for i, data in enumerate(dataloader, start=0):
            start_time = time.time()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            images = data['images'].to(device)
            labels = data['labels'].to(device)

            # prediction = net(images)
            # loss, dp, dn = criterion(prediction, labels)
            prediction = net(images)
            loss, dp, dn = criterion(prediction, labels, side_output=None)

            loss.backward()
            optimizer.step()

            # print statistics
            duration = time.time() - start_time
            print_epoch = 100
            running_loss += loss.item()
            
            if i % print_epoch == print_epoch - 1:  # print every 2000 mini-batches
                examples_per_sec = 10 / duration
                sec_per_batch = float(duration)
                format_str = '%s: step [%d, %5d/%4d], lr = %e, loss = %.3f (%.1f examples/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), epoch + 1, i + 1, len(dataloader), optimizer.param_groups[0]['lr'],
                                    running_loss / print_epoch, examples_per_sec, sec_per_batch))
                file_handle = open('1.txt', mode='a')
                file_handle.write(
                    format_str % (datetime.now(), epoch + 1, i + 1, len(dataloader), optimizer.param_groups[0]['lr'],
                                  running_loss / print_epoch, examples_per_sec, sec_per_batch))
                file_handle.write('\n')
                file_handle.close()
                running_loss = 0.0

            # validation
            validation_epoch = 100
            if not os.path.exists('./validation/'):
                os.makedirs('./validation/')
            if i % validation_epoch == validation_epoch - 1:
                # prediction, _, _ = net(images)
                # prediction = net(images)
                # prediction = torch.cat([prediction[0],prediction[1],prediction[2],prediction[3]],dim=3)
                prediction = prediction.cpu().detach().numpy().transpose((0, 2, 3, 1))
                for j in range(prediction.shape[0]):
                    cv2.imwrite('./validation/pred_' + str(j) + '.png', prediction[j] * 255)

                # ax = plt.subplot(1, 2, 1)
                # data_ = dp.cpu().detach().numpy()
                # ax.hist(data_, bins=np.linspace(0, 1, 100, endpoint=True))
                # ax = plt.subplot(1, 2, 2)
                # data_ = dn.cpu().detach().numpy()
                # ax.hist(data_, bins=np.linspace(0, 1, 100, endpoint=True))
                # plt.savefig('./validation/test' + str(epoch) + '.png')
                # plt.close('all')

        # save
        if not os.path.exists('./checkpiont/'):
            os.makedirs('./checkpiont/')
        save_epoch = str(epoch)
        torch.save(net.state_dict(), './checkpiont/' + save_epoch + cfgs['save_name'])
        state = {'epoch': epoch+1, 'param': net.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, 'checkpoint.pth')

    print('Finished Training')
    # os.system('shutdown -s -f -t 59')

