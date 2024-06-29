import argparse
import os, sys
import os
import pickle as pkl
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss, student_net
from torch.utils.data import DataLoader
# from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
# from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
# from image_source import load_data
from JL_modelevaluation import model_evaluation
from sklearn.metrics import classification_report
import datetime
import utils
import pandas as pd
import time

def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    data_train = pkl.load(open(os.path.join(args.data_dir, args.src_domain), 'rb'))
    data_target = pkl.load(open(os.path.join(args.data_dir, args.tgt_domain), 'rb'))
    data_test = pkl.load(open(os.path.join(args.data_dir, args.tgt_result), 'rb'))

    n_class = args.class_num
    source_loader, target_loader, test_loader = loader_get(data_train, data_target, data_test, args)
    return source_loader, target_loader, test_loader, n_class


def loader_get(data_train, data_target, data_test, args):
    feature_title = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12',
                     'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23', 'F24']

    source_loader = data_train[feature_title].astype(np.float32)
    target_loader = data_target[feature_title].astype(np.float32)
    test_loader = data_test[feature_title].astype(np.float32)

    source_label_loader = data_train['Label'].astype(np.int64)
    target_label_loader = data_target['Label'].astype(np.int64)
    test_label_loader = data_test['Label'].astype(np.int64)
    # Normalization
    # min_num = np.min(np.vstack([source_loader, target_test_loader]), axis=0)
    # max_num = np.max(np.vstack([source_loader, target_test_loader]), axis=0)
    min_num = source_loader.min(axis = 0)
    max_num = source_loader.max(axis = 0)
    source_loader = (source_loader - min_num) / (max_num - min_num)
    min_num = np.min(np.vstack([target_loader, test_loader]), axis=0)
    max_num = np.max(np.vstack([target_loader, test_loader]), axis=0)
    # min_num = np.min(np.vstack([source_loader, target_loader, test_loader]), axis=0)
    # max_num = np.max(np.vstack([source_loader, target_loader, test_loader]), axis=0)
    # min_num = target_loader.min(axis=0)
    # max_num = target_loader.max(axis=0)
    # target_train_loader = (target_train_loader - min_num) / (max_num - min_num)
    target_loader = (target_loader - min_num) / (max_num - min_num)
    test_loader = (test_loader - min_num) / (max_num - min_num)
    # Transfer to tensor format
    source_loader = torch.from_numpy(source_loader.values)
    target_loader = torch.from_numpy(target_loader.values)
    test_loader = torch.from_numpy(test_loader.values)

    source_label_loader = torch.from_numpy(source_label_loader.values)
    target_label_loader = torch.from_numpy(target_label_loader.values)
    test_label_loader = torch.from_numpy(test_label_loader.values)
    # Transfer to Batch-size type
    source_dataset = torch.utils.data.TensorDataset(source_loader, source_label_loader)
    source_loader = torch.utils.data.DataLoader(
        dataset=source_dataset,  # torch TensorDataset format
        batch_size=args.batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=1,  # 多线程来读数据
        drop_last=False,
    )

    target_dataset = torch.utils.data.TensorDataset(target_loader, target_label_loader)
    target_loader = torch.utils.data.DataLoader(
        dataset=target_dataset,  # torch TensorDataset format
        batch_size=args.batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=1,  # 多线程来读数据
        drop_last=False,
    )

    test_dataset = torch.utils.data.TensorDataset(test_loader, test_label_loader)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,  # torch TensorDataset format
        batch_size=args.batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=1,  # 多线程来读数据
        drop_last=False
    )
    return source_loader, target_loader, test_loader


def define_student_net(args):
    student_netF = student_net.StudentNetF(basenet_output = args.bottlenodes).to(args.device)
    student_netB = student_net.StudentNetB(basenet_output = args.bottlenodes, feature_dim=args.featurenodes).to(args.device)
    student_netC = student_net.StudentNetC(class_num=args.class_num, feature_dim=args.featurenodes).to(args.device)
    return student_netF, student_netB, student_netC


def get_model(args):
    netF = network.our_netF(basenet_list_output_num=args.bottlenodes).to(args.device)
    netB = network.our_netB(feature_dim=args.featurenodes, basenet_list_output_num=args.bottlenodes).to(args.device)
    netC = network.our_netC(class_num=args.class_num, feature_dim=args.featurenodes).to(args.device)
    return netF, netB, netC


def get_teacher_net(args):
    netF, netB, netC = get_model(args)
    modelpath = args.model_save + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.model_save + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.model_save + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netF.eval()
    netB.eval()
    netC.eval()
    return netF, netB, netC


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def get_student_optimizer(netF, netB,  netC, args):
    optimizer = torch.optim.SGD([
        {'params': netF.parameters(), 'lr': 1.0 * args.lr},
        {'params': netB.parameters(), 'lr': 1.0 * args.lr},
        {'params': netC.parameters(), 'lr': 1.0 * args.lr},
    ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    return optimizer


def get_model_evaluation(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        netF.eval()
        netB.eval()
        netC.eval()
        for data, labels in loader:
            data, labels = data.to(args.device), labels.to(args.device)
            feas = netB(netF(data))
            outputs = netC(feas)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)

    _, predict = torch.max(all_output, 1)
    predict = torch.squeeze(predict).float()
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    now_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = 'SFDA'
    model_evaluation(np.array(predict), np.array(all_label),  model_name, now_time)


def get_initial_model_evaluation(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        netF.eval()
        netB.eval()
        netC.eval()
        for data, labels in loader:
            data, labels = data.to(args.device), labels.to(args.device)
            feas = netB(netF(data))
            outputs = netC(feas)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)

    _, predict = torch.max(all_output, 1)
    predict = torch.squeeze(predict).float()
    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    print("---------- Evaluation on Test Data ----------" )
    class_result = classification_report(np.ravel(np.array(all_label)), np.array(predict), output_dict=True)
    print(classification_report(np.ravel(np.array(all_label)), np.array(predict), digits=4))



def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True

    with torch.no_grad():
        netF.eval()
        netB.eval()
        netC.eval()
        for data, labels in loader:
            data, labels = data.to(args.device), labels.to(args.device)
            feas = netB(netF(data))
            outputs = netC(feas)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    ValueF1Score = f1_score(np.ravel(np.array(all_label)), np.array(predict), average='macro')

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc, predict, ValueF1Score
    else:
        return accuracy*100, mean_ent, predict, ValueF1Score


def ADI_source_model(loader, netF, netB, netC):
    start_test = True

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(args.device), labels.to(args.device)
            feas = netB(netF(data))
            outputs = netC(feas)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)
    _, predict = torch.max(all_output, 1)
    return predict


def knowledge_distillation_loss(student_outputs, teacher_outputs, T):
    # alpha = 0.5
    return nn.KLDivLoss(reduction='batchmean')(nn.functional.log_softmax(student_outputs / T, dim=1), nn.functional.softmax(teacher_outputs / T, dim=1)) * (T * T)
           # * (T * T * alpha) \ + nn.CrossEntropyLoss()(outputs, labels) * (1. - alpha)



def train_target(netF, netB, netC, student_netF, student_netB, student_netC, optimizer, target_loader, test_loader, args):
    # target_loader, test_loader
    len_target_loader = len(target_loader)
    n_batch = len_target_loader
    best_acc = 0

    netF.eval()
    netB.eval()
    netC.eval()

    student_netF.eval()
    student_netB.eval()
    student_netC.eval()

    mem_label = obtain_label(target_loader, student_netF, student_netB, student_netC, args)
    mem_label = torch.from_numpy(mem_label).cuda()

    # model_predict_matrix = []

    for e in range(1, args.max_epoch + 1):
        loss_total = utils.AverageMeter()
        loss_pseudo = utils.AverageMeter()
        loss_clustering = utils.AverageMeter()
        loss_entropy = utils.AverageMeter()
        loss_regularization = utils.AverageMeter()

        if len_target_loader != 0:
            iter_target = iter(target_loader)
        # 定义自适应权重

        Weight_alpha = (1 - pow(2.71828, -args.weight_alpha_weight * e / args.max_epoch)) / (1 + pow(2.71828, -args.weight_alpha_weight * e / args.max_epoch))
        # Weight_alpha = 0.5
        Weight_Lambda1 = args.Weight_Lambda1 * (1 - args.W_L1 * Weight_alpha)
        Weight_Lambda2 = args.Weight_Lambda2 * Weight_alpha
        Weight_Lambda3 = args.Weight_Lambda3 * Weight_alpha
        Weight_Lambda4 = args.Weight_Lambda4 * Weight_alpha

        # if e >= 25:
        #     if e == 25:
        #         mem_label = obtain_label(target_loader, student_netF, student_netB, student_netC, args)
        #         mem_label = torch.from_numpy(mem_label).cuda()
        #         print('Add Self-Training with Pseudo Labels')
        #     elif e == 40:
        #         mem_label = obtain_label(target_loader, student_netF, student_netB, student_netC, args)
        #         mem_label = torch.from_numpy(mem_label).cuda()
        #         print('Add Self-Training with Pseudo Labels')
        if e % args.Updata_epoch_2 == 0:
            mem_label = obtain_label(target_loader, student_netF, student_netB, student_netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
        # elif e < 30:
        #     Weight_Lambda1 = args.Weight_Lambda1
        #     Weight_Lambda2 = 0
        #     Weight_Lambda3 = 0
        #     Weight_Lambda4 = 0
        # if e < 50:
        #
        #     WeightAdaptive = 0
        #     WeightKL = 3 - e / args.max_epoch
        # elif e == 50:
        #     WeightAdaptive = 1 * e / args.max_epoch
        #     mem_label = obtain_label(target_loader, student_netF, student_netB, student_netC, args)
        #     mem_label = torch.from_numpy(mem_label).cuda()
        #     print('Add Self-Training with Pseudo Labels')
        #     WeightKL = 3 - e / args.max_epoch
        # elif e == 70:
        #     WeightAdaptive = 1 * e / args.max_epoch
        #     mem_label = obtain_label(target_loader, student_netF, student_netB, student_netC, args)
        #     mem_label = torch.from_numpy(mem_label).cuda()
        #     print('Add Self-Training with Pseudo Labels')
        #     WeightKL = 3 - e / args.max_epoch
        # else:
        #     WeightAdaptive = 1 * e / args.max_epoch
        #     WeightKL = 3 - e / args.max_epoch

        # WeightKL = 0.5 #2 * (args.max_epoch - e) / args.max_epoch


        for ee in range(n_batch):

            student_netF.train()
            student_netB.train()
            student_netC.train()

            start_num = ee * args.batch_size
            end_num = (ee + 1) * args.batch_size
            data_target, labels_target = next(iter_target)
            data_target, labels_target = data_target.to(
                args.device), labels_target.to(args.device)

            # Add Noise
            if args.SNR != 1000:

                noise_tensor = torch.randn(data_target.shape[0], 24)
                noise_sum = torch.sum(torch.pow(noise_tensor, 2))
                feature_sum = torch.sum(torch.pow(data_target, 2))
                SNR_pow = np.power(10, args.SNR/10)
                SNR_ratio = torch.sqrt(feature_sum / (noise_sum * SNR_pow))
                # 调整高斯噪声的强度（方差）
                scaled_noise_tensor = noise_tensor.to(args.device) * SNR_ratio.to(args.device)
                scaled_noise_tensor = scaled_noise_tensor.to(args.device)
                # 将调整后的高斯噪声张量与目标张量相加
                data_target = data_target + scaled_noise_tensor

            outputs_target = student_netC(student_netB(student_netF(data_target)))
            outputs_soft_source = netC(netB(netF(data_target)))
            outputs_hard_source = torch.argmax(outputs_soft_source, -1)
            # outputs_hard_source = torch.nn.functional.one_hot(outputs_hard_source, num_classes = 4)

            # KL_temperature = 0.1
            # KL_soft_loss = knowledge_distillation_loss(outputs_target, outputs_soft_source, KL_temperature)
            outputs_hard_source = torch.Tensor(outputs_hard_source).long()
            # KL_hard_loss = nn.CrossEntropyLoss()(outputs_target, outputs_hard_source)
            # KL_loss = 0 * KL_soft_loss + KL_hard_loss
            Loss_pseudo = nn.CrossEntropyLoss()(outputs_target, outputs_hard_source)

            labels_predict = mem_label[start_num: end_num]
            classifier_loss = torch.nn.CrossEntropyLoss()(outputs_target, labels_predict.long())

            # 定义输出的自熵
            softmax_out = nn.Softmax(dim=1)(outputs_target)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
            # im_loss = Weight_Lambda3 * entropy_loss - Weight_Lambda4 * gentropy_loss

            # Adaptive_loss = Weight_Lambda2 * classifier_loss +  im_loss
            all_loss = Weight_Lambda1 * Loss_pseudo + Weight_Lambda2 * classifier_loss +  Weight_Lambda3 * entropy_loss - Weight_Lambda4 * gentropy_loss

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

            loss_total.update(all_loss.item())
            loss_pseudo.update(Loss_pseudo.item())
            loss_clustering.update(classifier_loss.item())
            loss_entropy.update(entropy_loss.item())
            loss_regularization.update(gentropy_loss.item())

        student_netF.eval()
        student_netB.eval()
        student_netC.eval()

        # acc_s_te, acc_list, model_predict_value, ValueF1Score = cal_acc(test_loader, student_netF, student_netB, student_netC, True)
        #
        # log_str = 'Iter: {}/{}; Accuracy = {:.2f}%, F1 Score = {:.4f}, L_total = {:.3f}, L_pseudo = {:.3f}, L_clustering = {:.3f}, L_entropy = {:.3f}, L_regularization = {:.3f}: '.format(e,
        #            args.max_epoch, acc_s_te, ValueF1Score, loss_total.avg, loss_pseudo.avg, loss_clustering.avg, loss_entropy.avg, loss_regularization.avg) + 'Acc-list: ' + acc_list + ', Weight_alpha = {:.3f}, Weight_Lambda1 = {:.3f}, Weight_Lambda2 = {:.3f}, Weight_Lambda3 = {:.3f}, Weight_Lambda4 = {:.3f}'.format(Weight_alpha,
        #            Weight_Lambda1, Weight_Lambda2, Weight_Lambda3, Weight_Lambda4)
        # print(log_str)
        # if e == args.epoch_stop_value:
        #     break

        # print('Weight_alpha = {:.3f}, Weight_Lambda1 = {:.3f}, Weight_Lambda2 = {:.3f}, Weight_Lambda3 = {:.3f}, Weight_Lambda4 = {:.3f}'.format(Weight_alpha,
        #            Weight_Lambda1, Weight_Lambda2, Weight_Lambda3, Weight_Lambda4)+ '\n')
        #if e == 9:
        #    torch.save(student_netF.state_dict(), os.path.join(args.model_save, "target_F_30" + ".pt"))
        #    torch.save(student_netB.state_dict(), os.path.join(args.model_save, "target_B_30" + ".pt"))
        #    torch.save(student_netC.state_dict(), os.path.join(args.model_save, "target_C_30" + ".pt"))
        #elif e == 11:
        #    torch.save(student_netF.state_dict(), os.path.join(args.model_save, "target_F_60" + ".pt"))
        #    torch.save(student_netB.state_dict(), os.path.join(args.model_save, "target_B_60" + ".pt"))
        #    torch.save(student_netC.state_dict(), os.path.join(args.model_save, "target_C_60" + ".pt"))
        #     print('get it accuracy = {:.3f}'.format(acc_s_te))

    # torch.save(student_netF.state_dict(), os.path.join(args.model_save, "target_F_" + ".pt"))
    # torch.save(student_netB.state_dict(), os.path.join(args.model_save, "target_B_" + ".pt"))
    # torch.save(student_netC.state_dict(), os.path.join(args.model_save, "target_C_" + ".pt"))
    # return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(args.device), labels.to(args.device)
            feas = netB(netF(data))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)

    ##
    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>args.threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    # log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    #
    # args.out_file.write(log_str + '\n')
    # args.out_file.flush()
    # print(log_str+'\n')
    return predict.astype('int')
def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def get_parser():
    "Get default arguments"
    parser = argparse.ArgumentParser(description='Source full-protection domain adaptation config parser')
    parser.add_argument('--max_epoch', type=int, default = 100, help="max iterations")
    parser.add_argument('--batch_size', type=int, default = 128, help="batch_size")
    parser.add_argument('--lr', type=float, default = 0.015, help="learning rate")
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=3409, help="random seed")
    parser.add_argument('--bottlenodes', type=int, default=100)
    parser.add_argument('--featurenodes', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--Updata_epoch_2', type=int, default=5)
    parser.add_argument('--W_L1', type=int, default=0.99)
    parser.add_argument('--Weight_Lambda1', type=int, default=5)
    parser.add_argument('--Weight_Lambda2', type=int, default=1)
    parser.add_argument('--Weight_Lambda3', type=int, default=0)
    parser.add_argument('--Weight_Lambda4', type=int, default=0)
    parser.add_argument('--weight_alpha_weight', type=int, default=5)
    parser.add_argument('--SNR', type=int, default= 1000)

    parser.add_argument('--epoch_stop_value', type=int, default=-1)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--class_num', type=int, default=4)
    parser.add_argument('--model_save', type=str, default='modelsave')
    parser.add_argument('--data_dir', type=str, default='Datasets')
    parser.add_argument('--src_domain', type=str, default='Datasets_C.pkl')
    parser.add_argument('--tgt_domain', type=str, default='Datasets_A.pkl')
    parser.add_argument('--tgt_result', type=str, default='Datasets_A.pkl')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    set_random_seed(args.seed)
    source_loader, target_loader, test_loader, n_class = load_data(args) #输入数据
    netF, netB, netC = get_teacher_net(args) #获取Teacher模型
    student_netF, student_netB, student_netC = define_student_net(args) # 定义学生模型
    optimizer = get_student_optimizer(student_netF, student_netB, student_netC, args) # 定义学生模型的优化器

    get_initial_model_evaluation(target_loader, netF, netB, netC, args)
    get_model_evaluation(test_loader, netF, netB, netC, args) # 先求解Source model在目标域的效果

    start_time = time.time()
    train_target(netF, netB, netC, student_netF, student_netB, student_netC, optimizer, target_loader, test_loader, args)
    # train_target(netF, netB, netC, student_netF, student_netB, student_netC, optimizer, test_loader, test_loader, args)
    end_time = time.time()
    print(end_time - start_time)
    get_model_evaluation(test_loader, student_netF, student_netB, student_netC, args)

