from __future__ import print_function, division

import sys
import time
import torch

from .util import AverageMeter, accuracy


def train_vanilla(epoch, train_loader, teacher, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    teacher = module_list[1]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)
        assert(len(input) % opt.ec_k == 0)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()
        
        # ===============================================
        parity_num = len(input) // opt.ec_k
        parities = torch.zeros((parity_num, input.shape[1], input.shape[2], input.shape[3]))
        for i in range(parity_num):
            parity = torch.zeros_like(input[0])
            for j in range(opt.ec_k):
                parity += (1 / opt.ec_k) * input[i*opt.ec_k+j]
            parities[i] = parity
        
        # ===============================================
        _, tmpinput = teacher(input)
        if torch.cuda.is_available():
                parities = parities.cuda()
        _, ptyinput = teacher(parities)

        # ===================forward=====================
        for i in range(opt.ec_k):
            preact = False
            for j in range(parity_num):
                if j == 0:
                    input = torch.cat((tmpinput[: i], 
                                    tmpinput[i + 1 : opt.ec_k], 
                                    ptyinput[0].unsqueeze(0)), 
                                    dim=0)
                else:
                    input = torch.cat((input,
                                    tmpinput[j * opt.ec_k : j * opt.ec_k + i], 
                                    tmpinput[j * opt.ec_k + i + 1 : (j+1) * opt.ec_k], 
                                    ptyinput[j].unsqueeze(0)),
                                        dim=0)
            
            if opt.distill in ['abound']:
                preact = True
            if not opt.irev:
                feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
            else:
                if opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
                    logit_s, _ = model_s(input.reshape((-1, 8*opt.ec_k, 64, 64)))
                elif opt.dataset == 'mnist' or opt.dataset == 'fashion':
                    logit_s, _ = model_s(input.reshape((-1, 8*opt.ec_k, 56, 56)))
                elif opt.dataset == 'stl10':
                    logit_s, _ = model_s(input.reshape((-1, 2*opt.ec_k, 384, 384)))
                elif opt.dataset == 'speech':
                    logit_s, _ = model_s(input.reshape((-1, 5*opt.ec_k, 320, 320)))
                else:
                    logit_s, _ = model_s(input)
            with torch.no_grad():
                if not opt.irev:
                    feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
                    feat_t = [f.detach() for f in feat_t]
                else:
                    logit_t = model_t(input)

            tmp_targets = torch.index_select(target.cpu(), dim=0, index=torch.tensor([x for x in range(i, len(target), opt.ec_k)]))
            if torch.cuda.is_available():
                tmp_targets = tmp_targets.cuda()
            # cls + kl div
            loss_cls = criterion_cls(logit_s, tmp_targets)
            loss_div = criterion_div(logit_s, logit_t)

            # other kd beyond KL divergence
            if opt.distill == 'kd':
                loss_kd = 0
            elif opt.distill == 'hint':
                f_s = module_list[1](feat_s[opt.hint_layer])
                f_t = feat_t[opt.hint_layer]
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'crd':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
            elif opt.distill == 'attention':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'nst':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'similarity':
                g_s = [feat_s[-2]]
                g_t = [feat_t[-2]]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'rkd':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'pkt':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'kdsvd':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'correlation':
                f_s = module_list[1](feat_s[-1])
                f_t = module_list[2](feat_t[-1])
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'vid':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
                loss_kd = sum(loss_group)
            elif opt.distill == 'abound':
                # can also add loss to this stage
                loss_kd = 0
            elif opt.distill == 'fsp':
                # can also add loss to this stage
                loss_kd = 0
            elif opt.distill == 'factor':
                factor_s = module_list[1](feat_s[-2])
                factor_t = module_list[2](feat_t[-2], is_factor=True)
                loss_kd = criterion_kd(factor_s, factor_t)
            else:
                raise NotImplementedError(opt.distill)

            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

            acc1, acc5 = accuracy(logit_s, tmp_targets, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def validate(val_loader, model_t, model_s, criterion, opt, teacher=False):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model_s.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (data, target) in enumerate(val_loader):
            data = data.float()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            # ========================================
            parity_num = len(data) // opt.ec_k
            parities = torch.zeros((parity_num, data.shape[1], data.shape[2], data.shape[3]))
            for i in range(parity_num):
                parity = torch.zeros_like(data[0])
                for j in range(opt.ec_k):
                    parity += (1 / opt.ec_k) * data[i*opt.ec_k + j]
                parities[i] = parity
            
            # ========================================
            _, tmpinput = model_t(data)
            if torch.cuda.is_available():
                parities = parities.cuda()
            _, ptyinput = model_t(parities)
            
            for i in range(opt.ec_k):
                for j in range(parity_num):
                    if j == 0:
                        input = torch.cat((tmpinput[:i], 
                                        tmpinput[i + 1 : opt.ec_k], 
                                        ptyinput[0].unsqueeze(0)), 
                                        dim=0)
                    else:
                        input = torch.cat((input,
                                        tmpinput[j * opt.ec_k : j * opt.ec_k + i], 
                                        tmpinput[j * opt.ec_k + i + 1 : (j+1) * opt.ec_k], 
                                        ptyinput[j].unsqueeze(0)),
                                          dim=0)
                # compute output
                if opt.irev:
                    if teacher:
                        output = model_s(input)
                    else:
                        if opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
                            output, _ = model_s(input.reshape((-1, 8*opt.ec_k, 64, 64)))
                        elif opt.dataset == 'mnist' or opt.dataset == 'fashion':
                            output, _ = model_s(input.reshape((-1, 8*opt.ec_k, 56, 56)))
                        elif opt.dataset == 'stl10':
                            output, _ = model_s(input.reshape((-1, 2*opt.ec_k, 384, 384)))
                        elif opt.dataset == 'speech':
                            output, _ = model_s(input.reshape((-1, 5*opt.ec_k, 320, 320)))
                        else:
                            output, _ = model_s(input)
                else:
                    output = model_s(input)
                
                tmp_targets = torch.index_select(target.cpu(), dim=0, index=torch.tensor([x for x in range(i, len(target), opt.ec_k)]))
                if torch.cuda.is_available():
                    tmp_targets = tmp_targets.cuda()
                # print(output.shape)
                # print(tmp_targets.shape)
                loss = criterion(output, tmp_targets)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, tmp_targets, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
