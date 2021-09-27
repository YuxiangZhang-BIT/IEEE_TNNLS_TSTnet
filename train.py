from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from model import TSTnet
import numpy as np
from utils_HSI import sample_gt, metrics, get_device, seed_worker
from datasets import get_dataset, HyperX, data_prefetcher
from datetime import datetime
import os
import torch.utils.data as data
from tensorboardX import SummaryWriter
import scipy.io as io
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='PyTorch TSTnet')

parser.add_argument('--save_path', type=str, default="./results/",
                    help='the path to save the model')
parser.add_argument('--data_path', type=str, default='./datasets/HyRank/',
                    help='the path to load the data')
parser.add_argument('--log_path', type=str, default='./logs',
                    help='the path to load the data')
parser.add_argument('--source_name', type=str, default='Dioni',
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='Loukia',
                    help='the name of the test dir')
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")

# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--patch_size', type=int, default=12,
                    help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")
group_train.add_argument('--lr', type=float, default=1e-2,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--lambda_1', type=float, default=1e+0,
                    help="Regularization parameter, controlling the contribution of distribution alignment.")
group_train.add_argument('--lambda_2', type=float, default=1e-1,
                    help="Regularization parameter, controlling the contribution of graph alignment.")
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
group_train.add_argument('--batch_size', type=int, default=100,
                    help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--test_stride', type=int, default=1,
                     help="Sliding window step stride during inference (default = 1)")
parser.add_argument('--seed', type=int, default=1233, metavar='S',
                    help='random seed ')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=1e-4,
                    help='the L2  weight decay')

parser.add_argument('--num_epoch', type=int, default=500,
                    help='the number of epoch')
parser.add_argument('--num_trials', type=int, default=10,
                    help='the number of epoch')
parser.add_argument('--training_sample_ratio', type=float, default=0.05,
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=10,
                    help='multiple of of data augmentation')

# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=True,
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true',default=True,
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true',default=False,
                    help="Random mixes between spectra")

args = parser.parse_args()
DEVICE = get_device(args.cuda)

def train(epoch, model_TST, num_epoch):

    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch) / num_epoch), 0.75)
    optimizer_TST = optim.SGD(model_TST.parameters(), lr=LEARNING_RATE, momentum=args.momentum,weight_decay = args.l2_decay)

    if (epoch-1)%10==0:
        print('learning rate{: .4f}'.format(LEARNING_RATE) )

    global writer
    CNN_cls_loss, mmd_loss, got_loss, GCN_cls_loss, Entropy_loss = 0, 0, 0, 0, 0
    CNN_correct, CNN_tar_correct, GCN_correct = 0, 0, 0
    len_tar_temp = 0

    iter_source = data_prefetcher(train_loader)
    iter_target = data_prefetcher(train_tar_loader)
    num_iter = len_src_loader
    bs = train_loader.batch_size

    for i in range(1, num_iter):

        model_TST.train()
        if 0 < (len_tar_train_dataset-i*bs) < bs or i % len_tar_train_loader == 0:
            iter_target = data_prefetcher(train_tar_loader)
        data_src, label_src = iter_source.next()
        data_tar, label_tar = iter_target.next()
        label_src = label_src - 1
        label_tar = label_tar - 1

        optimizer_TST.zero_grad()
        out = model_TST(data_src, data_tar)
        label_src_pred, loss_mmd, TST_wd, TST_gwd = out[0], out[1], out[2], out[3]
        label_tar_pred, label_src_gcn_pred = out[4], out[5]

        # classification loss
        loss_cls = F.nll_loss(F.log_softmax(label_src_pred, dim=1), label_src.long())
        loss_gcn_cls = F.nll_loss(F.log_softmax(label_src_gcn_pred, dim=1), label_src.long())

        # consistency loss
        entropy_tar = -sum(sum(torch.mul(F.softmax(label_src_pred, dim=1),F.log_softmax(label_src_gcn_pred, dim=1))))/label_tar_pred.shape[0]

        # GOT loss
        loss_got = TST_wd + TST_gwd

        # total loss
        loss_TST = loss_cls + loss_gcn_cls + args.lambda_1*loss_mmd + args.lambda_2*loss_got + entropy_tar

        CNN_cls_loss += loss_cls.item()
        GCN_cls_loss += loss_gcn_cls.item()
        mmd_loss += loss_mmd.item()
        got_loss += loss_got.item()
        Entropy_loss += entropy_tar.item()
        loss_TST.backward()
        optimizer_TST.step()

        pred = label_src_pred.data.max(1)[1] 
        CNN_correct += pred.eq(label_src.data.view_as(pred)).cpu().sum()
        pred_gcn = label_src_gcn_pred.data.max(1)[1]
        GCN_correct += pred_gcn.eq(label_src.data.view_as(pred_gcn)).cpu().sum()
        pred_tar = label_tar_pred.data.max(1)[1]
        len_tar_temp += len(pred_tar)

        if len_tar_train_dataset - len_tar_temp >= 0:
            CNN_tar_correct += pred_tar.eq(label_tar.data.view_as(pred_tar)).cpu().sum()
            len_tar = len_tar_temp

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format( epoch+1, i * len(data_src), len_src_dataset, 100. * i / len_src_loader))
            print('loss_TSTnet: {:.6f},  loss_cls: {:.6f},  loss_gcn_cls: {:.6f}, loss_mmd: {:.6f}, loss_got: {:.6f}'.format(
            loss_TST.item(), loss_cls.item(), loss_gcn_cls.item(), loss_mmd.item(), loss_got))

    CCN_acc = CNN_correct.item() / len_src_dataset
    GCN_acc = GCN_correct.item() / len_src_dataset
    CCN_tar_acc = CNN_tar_correct.item() / len_tar

    print('[epoch: {:4}]  Train Accuracy: {:.4f} | train sample number: {:6}'.format(epoch+1, CCN_acc, len_src_dataset))
    writer.add_scalars('CNN_Loss_group', {'CNN_cls_loss': CNN_cls_loss/len_src_loader, 'GCN_cls_loss':GCN_cls_loss/len_src_loader ,'mmd_loss': mmd_loss/len_src_loader,
                        'got_loss': got_loss/len_src_loader, 'Entropy_loss': Entropy_loss/len_src_loader }, epoch)

    return model_TST, CCN_acc, CCN_tar_acc, GCN_acc

def test(model):
    model.eval()
    loss = 0
    correct = 0
    pred_list, label_list = [], []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            label = label - 1
            out = model_TST(data, label)
            pred = out[0].data.max(1)[1]
            pred_list.append(pred.cpu().numpy())
            label_list.append(label.cpu().numpy())
            loss += F.nll_loss(F.log_softmax(out[0], dim = 1), label.long()).item() # sum up batch loss
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()

        loss /= len_tar_loader
        print('Testing...')
        print('{} set: Average test loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n, | Test sample number: {:6}'.format(
            args.target_name, loss, correct, len_tar_dataset,
            100. * correct / len_tar_dataset, len_tar_dataset))
    return correct, correct.item() / len_tar_dataset, pred_list, label_list

if __name__ == '__main__':

    seed_worker(args.seed)

    acc_test_list = np.zeros([args.num_trials,1])
    for flag in range(args.num_trials):

        img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                                args.data_path)
        img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                                args.data_path)
        sample_num_src = len(np.nonzero(gt_src)[0])
        sample_num_tar = len(np.nonzero(gt_tar)[0])

        tmp = args.training_sample_ratio*args.re_ratio*sample_num_src/sample_num_tar
        training_sample_tar_ratio = tmp if tmp < 1 else 1

        num_classes=gt_src.max()
        N_BANDS = img_src.shape[-1]
        hyperparams = vars(args)
        hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 
                            'device': DEVICE, 'center_pixel': False, 'supervision': 'full'})
        hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

        r = int(hyperparams['patch_size']/2)+1
        img_src=np.pad(img_src,((r,r),(r,r),(0,0)),'symmetric')
        img_tar=np.pad(img_tar,((r,r),(r,r),(0,0)),'symmetric')
        gt_src=np.pad(gt_src,((r,r),(r,r)),'constant',constant_values=(0,0))
        gt_tar=np.pad(gt_tar,((r,r),(r,r)),'constant',constant_values=(0,0))     

        train_gt_src, val_gt_src, training_set, valing_set = sample_gt(gt_src, args.training_sample_ratio, mode='random')
        test_gt_tar, _, tesing_set, _ = sample_gt(gt_tar, 1, mode='random')
        train_gt_tar, _, _, _ = sample_gt(gt_tar, training_sample_tar_ratio, mode='random')
        img_src_con, img_tar_con, train_gt_src_con, train_gt_tar_con = img_src, img_tar, train_gt_src, train_gt_tar
        val_gt_src_con = val_gt_src
        if tmp < 1:
            for i in range(args.re_ratio-1):
                img_src_con = np.concatenate((img_src_con,img_src))
                train_gt_src_con = np.concatenate((train_gt_src_con,train_gt_src))
                val_gt_src_con = np.concatenate((val_gt_src_con,val_gt_src))
                # img_tar_con = np.concatenate((img_tar_con,img_tar))
                # train_gt_tar_con = np.concatenate((train_gt_tar_con,train_gt_tar))
        
        # Generate the dataset
        hyperparams_train = hyperparams.copy()

        train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
        g = torch.Generator()
        g.manual_seed(args.seed)
        train_loader = data.DataLoader(train_dataset,
                                        batch_size=hyperparams['batch_size'],
                                        pin_memory=True,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        shuffle=True)
        val_dataset = HyperX(img_src_con, val_gt_src_con, **hyperparams)
        val_loader = data.DataLoader(val_dataset,
                                        pin_memory=True,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        batch_size=hyperparams['batch_size'])
        train_tar_dataset = HyperX(img_tar_con, train_gt_tar_con, **hyperparams)
        train_tar_loader = data.DataLoader(train_tar_dataset,
                                        pin_memory=True,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        batch_size=hyperparams['batch_size'],
                                        shuffle=True)     
        test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
        test_loader = data.DataLoader(test_dataset,
                                        pin_memory=True,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        batch_size=hyperparams['batch_size'])                          
        len_src_loader = len(train_loader)
        len_tar_train_loader = len(train_tar_loader)
        len_src_dataset = len(train_loader.dataset)
        len_tar_train_dataset = len(train_tar_loader.dataset)
        len_tar_dataset = len(test_loader.dataset)
        len_tar_loader = len(test_loader)
        len_val_dataset = len(val_loader.dataset)
        len_val_loader = len(val_loader)
        print(hyperparams)
        print("train samples :",len_src_dataset)
        print("train tar samples :",len_tar_train_dataset)

        correct, val_correct, val_max_test, acc, val_loss = 0, 0, 0, 0, 10
        model_TST = TSTnet.Feature_Extractor(img_src.shape[-1],num_classes=gt_src.max(), patch_size=hyperparams['patch_size']).to(DEVICE)

        now_time = datetime.now()
        time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
        log_dir = os.path.join(args.log_path, args.source_name+'_'+time_str+'_lr_'+str(args.lr)+'_lam1_'+str(args.lambda_1)+'_lam2_'+str(args.lambda_2))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)

        for epoch in range(args.num_epoch):
            model_TST, CCN_train_acc, CCN_tar_acc, GCN_acc = train(epoch, model_TST, args.num_epoch)
        
            if epoch % args.log_interval == 0:
                t_correct, CCN_test_acc, pred, label = test(model_TST)
                if t_correct > correct:
                    correct = t_correct
                    acc = CCN_test_acc
                    if acc > 0.5:
                        acc_test_list[flag] = acc
                        results = metrics(np.concatenate(pred), np.concatenate(label), ignored_labels=hyperparams['ignored_labels'], n_classes=gt_src.max())
                        print(classification_report(np.concatenate(pred),np.concatenate(label),target_names=LABEL_VALUES_tar))
                        model_save_path = os.path.join(args.save_path, 'TSTnet_params_'+args.source_name+'_'+str(int(acc*100))+'.pkl')
                        torch.save(model_TST.state_dict(), model_save_path)

            print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
                args.source_name, args.target_name, correct, 100. * correct / len_tar_dataset ))

            writer.add_scalars('Accuracy_group', {'CCN_train_acc': CCN_train_acc, 'GCN_acc': GCN_acc, 'CCN_tar_acc': CCN_tar_acc, 'CCN_test_acc': CCN_test_acc}, epoch)
        io.savemat(os.path.join(args.save_path,'results_'+str(int(flag+1))+'times_'+args.source_name+'.mat'), {'results': results})
        io.savemat(os.path.join(args.save_path,'train_times_'+args.source_name+'.mat'), {'acc_test_list': acc_test_list,'lr':args.lr,'lambda1':args.lambda_1,'lambda2':args.lambda_2})
