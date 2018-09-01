import magnet as mag

from magnet.utils.images import show_images

import torch

@mag.eval
def order(model, data, c=10, n=5):
    x, y = next(iter(data(mode='val')))
    channels, rows, cols = x.shape[1:]
    images = torch.zeros(n, c, channels, rows, cols)

    images_gathered = [0] * c
    correct = 0
    dl = iter(data(shuffle=True, mode='val'))
    for i in range(len(dl)):
        x, y_true = next(dl)
        y = model(x)[0].max(0)[1].item()

        n_y = images_gathered[y]
        if n_y >= n: continue

        if y == y_true.item(): correct += 1
        else: x *= -1
        images[n_y, y] = x[0]
        images_gathered[y] += 1

    show_images(images.view(-1, 1, rows, cols), pixel_range=(-1, 1),
                titles=f'{int(correct * 100 / (c * n))}%')

def visualize_weights(model):
    show_images(model.layer.weight.view(10, -1, 28, 28), cmap='seismic')