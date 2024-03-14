import torch
from jitfields import set_num_threads


def get_test_devices():
    devices = [('cpu', 1), ('cpu', 4)]
    if torch.cuda.is_available():
        print('cuda backend available')
        devices.append('cuda')
    return devices


def init_device(device):
    if isinstance(device, (list, tuple)):
        device, param = device
    else:
        param = 1 if device == 'cpu' else 0
    if device == 'cuda':
        torch.cuda.set_device(param)
        torch.cuda.init()
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
        device = '{}:{}'.format(device, param)
    else:
        assert device == 'cpu'
        set_num_threads(param)
    device = torch.device(device)
    return device
