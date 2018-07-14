from torch.nn  import functional as F
from magnet.functional import activations, losses, metrics

dimensional_function = lambda f_list, *args, **kwargs: f_list[len(args[0].size()) - 3](*args, **kwargs)

adaptive_avg_pool = lambda x, output_size: dimensional_function([F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_avg_pool3d], x, output_size)

global_avg_pool = lambda x: adaptive_avg_pool(x, 1).squeeze()

wiki = {'global_avg_pool': global_avg_pool, 'activations': activations.wiki, 'losses': losses.wiki, 'metrics': metrics.wiki}