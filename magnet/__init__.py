from .autograd import eval, device, build_lock

def __print_init_message():
	from torch.cuda import get_device_name
	if device == 'cuda': print('Accelerating your code on shiney new', get_device_name(0), 'GPU.')
	else: print("Running your code on a slow, boring CPU.\nMake some money and buy yourself a GPU, will ya?"
            "\n\nPro Tip: If you're a poor old hag like me, use a cloud provider.")

__print_init_message()
