import torch
import torch.nn.functional as F
import time
import os
import sys
import numpy as np
from imageio import imwrite

from utils import AverageMeter

def normalize_data(data):
    data_min = np.min(data)
    data_max = np.max(data)
    data_norm = np.clip((data - data_min) *
                        (255.0 / (data_max - data_min)),
                0, 255).astype(np.uint8)
    return data_norm

def save_video_results(output_buffer, save_path):
    video_outputs = torch.stack(output_buffer)
    for i in range(video_outputs.size(0)):
        save_name = os.path.join(save_path, 'pred_sal_{0:06d}.jpg'.format(i+9))
        imwrite(save_name, normalize_data(video_outputs[i][0].data.numpy()))


def test(data_loader, model, opt):
    print('test')

    model.eval()

    with torch.no_grad():

        batch_time = AverageMeter()
        data_time = AverageMeter()

        end_time = time.time()
        output_buffer = []
        previous_video_id = ''

        for i, (data, targets, valid) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            inputs = data['rgb']
            curr_batch_size = inputs.size(0)

            while inputs.size(0) < opt.batch_size:
                inputs = torch.cat((inputs, inputs[0:1, :]), 0)
            while data['audio'].size(0) < opt.batch_size:
                data['audio'] = torch.cat((data['audio'], data['audio'][0:1, :]), 0)

            outputs = model(inputs, data['audio'])

            for ii in range(0, len(outputs['sal'])):
                outputs['sal'][ii] = outputs['sal'][ii][0:curr_batch_size,:]

            if not opt.no_sigmoid_in_test:
                outputs['sal'] = F.sigmoid(outputs['sal'][-1])

            for j in range(outputs['sal'].size(0)):
                if not (i == 0 and j == 0) and targets['video'][j] != previous_video_id:
                    save_path = os.path.join(opt.result_path, opt.dataset, previous_video_id)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_video_results(output_buffer, save_path)
                    output_buffer = []
                output_buffer.append(outputs['sal'][j].data.cpu())
                previous_video_id = targets['video'][j]

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time))

    save_path = os.path.join(opt.result_path, opt.dataset, previous_video_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_video_results(output_buffer, save_path)

