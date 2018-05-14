from .autograd import eval, device

if device == 'gpu': print('Accelerating your code on shiney new', torch.cuda.get_device_name(0), 'GPU.')
else: print("Running your code on a slow, boring CPU.\nMake some money and buy yourself a GPU, will ya?"
            "\n\nPro Tip: If you're a poor old hag like me, use a cloud provider.")