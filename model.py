import torch
from torch import nn
from models import resnet as resnet


def generate_model(opt):

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet import get_fine_tuning_parameters

        if opt.model_depth == 10:
            model = resnet.resnet10(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                audiovisual=opt.audiovisual)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                audiovisual=opt.audiovisual)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                audiovisual=opt.audiovisual)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                audiovisual=opt.audiovisual)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                audiovisual=opt.audiovisual)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                audiovisual=opt.audiovisual)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                audiovisual=opt.audiovisual)
    else:
        raise ValueError('Model type not supported')

    if not opt.no_cuda:
        model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)

    if opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path, map_location=lambda storage, loc: storage)
        assert opt.arch == pretrain['arch']
        model.load_state_dict(pretrain['state_dict'], strict=0)

    if opt.audio_pretrain_path:
        print('loading pretrained audio model {}'.format(opt.audio_pretrain_path))
        pretrain_audio = torch.load(opt.audio_pretrain_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(pretrain_audio, strict=0)

    (parameters, name_parameters) = get_fine_tuning_parameters(model, opt.learning_rate_sal, opt.weight_decay)
    
    return model, parameters
