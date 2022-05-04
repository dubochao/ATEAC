import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import Optimizer
from log import *
import numpy as np
writer = SummaryWriter('./net0/runs')
grad_clip = 1.0  # clip gradients at an absolute value of
save_prefix = ''
# import mxnet.gluon.loss as gloss
from sklearn.metrics import f1_score
gpuid=0

def clip_gradient(optimizer, grad_clip):
    #     """
    # 剪辑反向传播期间计算的梯度，以避免梯度爆炸。
    #
    # param optimizer：具有要剪裁的渐变的优化器
    #
    # ：参数梯度剪辑：剪辑值
    #     """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train(train_iter, dev_iter, model, dir, args):
    # global args

    global save_prefix
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = dir +'_'+ args.snapshot
    save_prefix = os.path.join(save_dir, filename)
    if os.path.exists(save_prefix):
        logging.info('\nLoading model from {}...\n'.format(save_prefix))
        model = torch.load(save_prefix)['model']
        # optimizer = torch.load(snapshot)['optimizer']
    else:
        print('\nRunning model {}...\n'.format(dir))
    if args.Optimizer:
        optimizer = Optimizer.TransformerOptimizer(
            torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001,, betas=(0.9, 0.98), eps=1e-09)
    if args.cuda:
        print('model.cuda(gpuid)==',gpuid)
        model.cuda(gpuid)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    best_f1 = 0
    last_step = 0
    for epoch in range(1, args.epochs + 1):

        for batch in train_iter:

            model.train()
            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)
            # w.add_graph(model, (feature,))
            if args.cuda:
                feature, target = feature.cuda(gpuid), target.cuda(gpuid)
            optimizer.zero_grad()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            # Clip gradients
            if args.Optimizer:
                clip_gradient(optimizer.optimizer, grad_clip)
                lr = optimizer.lr
            else:
                lr = optimizer.state_dict()['param_groups'][0]['lr']

            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                y_pred = torch.max(logits, 1)[1].view(target.size()).data
                corrects = (y_pred == target.data).sum()
                train_acc = corrects / batch.batch_size
                F1 = f1_score(target.data.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f} F1: {:.4f} acc: {:.4f}({}/{})'.format(steps,
                                                                                      loss.item(),
                                                                                      F1,
                                                                                      train_acc,
                                                                                      corrects,
                                                                                      batch.batch_size))

                writer.add_scalars('train_learning_rate', {dir + '_lr': lr},steps)
                # writer.add_scalars('train_loss', {dir + '_loss': loss.item()},steps)
                # writer.add_scalars('run_train_F1', {dir + '_F1': F1}, steps)
                # writer.add_scalars('run_train_accuracy', {dir + '_accuracy': train_acc}, steps)
            if steps % args.test_interval == 0:
                dev_acc, dev_f1 = eval(dev_iter, model, args, steps, dir)

                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    optimizer.stop_up=0
                    if args.save_best:
                        sys.stdout.write('Saving best model, acc: {:.4f}\n'.format(best_acc))
                        save(model, best_acc, optimizer)

                if dev_f1 > best_f1:
                    best_f1 = dev_f1
                    last_step = steps
                    optimizer.stop_up = 0

                    if args.save_best:
                        sys.stdout.write('best model, best_f1: {:.4f}\n'.format(best_f1))

                #         args.early_stopping 这么多次未提升则停止
                if steps - last_step >= args.early_stopping:
                    optimizer.stop_up = 1

        writer.add_scalars('best_f1', {dir + '_best_f1': best_f1}, epoch)
        writer.add_scalars('best_acc', {dir + '_best_acc': best_acc}, epoch)
    sys.stdout.write(
        '\n{}  stop by {} steps, acc: {:.4f} best_f1: {:.4f}\n'.format(dir, epoch,
                                                                          best_acc, best_f1))

def eval(data_iter, model, args, steps, dir):
    model.eval()
    corrects, F1scores, F1size, avg_loss = 0, 0, 0, 0
    for batch in data_iter:

        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)
        # feature.t_(), target.sub_(1)
        if args.cuda:
            feature, target = feature.cuda(gpuid), target.cuda(gpuid)
        logits = model(feature)
        # loss = gloss.SoftmaxCrossEntropyLoss()
        # loss =loss(logits, target)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()

        F1size += 1
        F1scores += f1_score(target.data.cpu().numpy(), torch.max(logits, 1)[1].view(target.size()).data.cpu().numpy(),
                             average='macro')
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = corrects / size
    F1scores /= F1size
    # Recall = F1scores / F1size
    # F1=2 * Recall * accuracy / (Recall+accuracy)
    sys.stdout.write('\nEvaluation - loss: {:.6f}  F1: {:.4f} acc: {:.4f}({}/{}) \n'.format(avg_loss, F1scores,
                                                                                            accuracy,
                                                                                            corrects,
                                                                                            size))
    return accuracy, F1scores


def save(model, best_acc, optimizer):
    state = {
        'best_acc': best_acc,
        'model': model,
        'optimizer': optimizer}

    torch.save(state, save_prefix)
