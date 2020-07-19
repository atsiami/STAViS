import torch
import torch.nn.functional as F
import time
import sys
from numpy import nonzero

from utils import AverageMeter
from models.sal_losses import cross_entropy_loss, cc_score, nss_score


def val_epoch(epoch, nEpochs, data_loader, model, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    with torch.no_grad():

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_out = {'global': AverageMeter(), 'sal': AverageMeter()}
        losses = AverageMeter()
        sal_cross = AverageMeter()
        cc = AverageMeter()
        nss = AverageMeter()

        end_time = time.time()
        for i, (data, targets, valid) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            if not opt.no_cuda:
                targets['salmap'] = targets['salmap'].cuda()
                targets['binmap'] = targets['binmap'].cuda()
                valid['sal'] = valid['sal'].cuda()

            inputs = data['rgb']
            targets['salmap'] = targets['salmap'].float()
            targets['binmap'] = targets['binmap'].float()
            valid['sal'] = valid['sal'].float()

            curr_batch_size = inputs.size(0)

            while inputs.size(0) < opt.batch_size:
                inputs = torch.cat((inputs, inputs[0:1, :]), 0)

            outputs = model(inputs, data['audio'])

            for ii in range(0, len(outputs['sal'])):
                outputs['sal'][ii] = outputs['sal'][ii][0:curr_batch_size, :]

            loss = {'sal': []}

            sal_losses_BCE = [0] * len(outputs['sal'])
            sal_losses_CC = [0] * len(outputs['sal'])
            sal_losses_NSS = [0] * len(outputs['sal'])
            for ii in range(0, len(outputs['sal'])):
                sal_losses_BCE[ii] = cross_entropy_loss(outputs['sal'][ii], targets['salmap'], valid['sal'])
                sal_losses_CC[ii] = cc_score(outputs['sal'][ii], targets['salmap'], valid['sal'])
                sal_losses_NSS[ii] = nss_score(outputs['sal'][ii], targets['binmap'], valid['sal'])
            loss['sal'].append((1 - epoch / nEpochs) * sum(sal_losses_BCE[:-1]) + sal_losses_BCE[-1])
            loss['sal'].append((1 - epoch / nEpochs) * sum(sal_losses_CC[:-1]) + sal_losses_CC[-1])
            loss['sal'].append((1 - epoch / nEpochs) * sum(sal_losses_NSS[:-1]) + sal_losses_NSS[-1])

            loss['sal_total'] = opt.sal_weights[0] * loss['sal'][0] + \
                                opt.sal_weights[1] * loss['sal'][1] + \
                                opt.sal_weights[2] * loss['sal'][2]

            loss_all = loss['sal_total'] / opt.batch_sizes['sal']

            loss_all_tmp = {'global': 0, 'sal': 0}

            if sum(valid['sal']) > 0:
                cc_tmp = sal_losses_CC[-1].data / nonzero(valid['sal'])[:, 0].size(0)
                nss_tmp = sal_losses_NSS[-1].data / nonzero(valid['sal'])[:, 0].size(0)
                sal_cross_tmp = torch.sum(sal_losses_BCE[-1]) / nonzero(valid['sal'])[:, 0].size(0)
                cc.update(cc_tmp, nonzero(valid['sal'])[:, 0].size(0))
                nss.update(nss_tmp, nonzero(valid['sal'])[:, 0].size(0))
                sal_cross.update(sal_cross_tmp, nonzero(valid['sal'])[:, 0].size(0))
                loss_all_tmp['sal'] = opt.sal_weights[0] * sal_losses_BCE[-1] + \
                                      opt.sal_weights[1] * sal_losses_CC[-1] + \
                                      opt.sal_weights[2] * sal_losses_NSS[-1]
                loss_all_tmp['sal'] = loss_all_tmp['sal'] / nonzero(valid['sal'])[:, 0].size(0)
                losses_out['sal'].update(loss_all_tmp['sal'].data, nonzero(valid['sal'])[:, 0].size(0))

            loss_all_tmp['global'] = loss_all_tmp['sal']

            losses.update(loss_all.data[0], inputs.size(0))
            losses_out['global'].update(loss_all_tmp['global'].data, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'CC {cc.val:.3f} ({cc.avg:.3f})\t'
                  'NSS {nss.val:.3f} ({nss.avg:.3f})'.format(
                epoch,
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                cc=cc,
                nss=nss))

        logger.log({
            'epoch': epoch,
            'loss': losses_out['global'].avg,
            'loss_sal': losses_out['sal'].avg,
            'sal_cross': sal_cross.avg,
            'cc': cc.avg,
            'nss': nss.avg
        })

        return losses_out
