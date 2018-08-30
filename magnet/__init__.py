from ._autograd import eval, device, build_lock

def __print_init_message():
    from magnet.utils.misc import in_notebook

    if not in_notebook:
        print('MagNet Inside')
        return

    from torch.cuda import get_device_name
    if device.type == 'cpu':
        print("Running your code on a boring CPU.")
    else:
        print('Accelerating your code on a shiney new', get_device_name(0), '.')

__print_init_message()
