import torch
import numpy as np

from magnet.utils.varseq import pack, unpack

def test_pack_unpack():
    x = [torch.arange(i) for i in range(1, 6)]
    x_packed = pack(x)

    assert all(torch.all(x_unpacked_i[:len(x_i)] == x_i)
               for x_unpacked_i, x_i in zip(unpack(x_packed)[0].t(), x))

    assert all(torch.all(x_unpacked_i == x_i)
               for x_unpacked_i, x_i in zip(unpack(x_packed, as_list=True), x))

def test_pack_padded():
    x = torch.zeros(6, 6)
    for i in range(6): x[i, i:] = i

    x_packed = pack(x, lengths=np.arange(1, 7))
    assert torch.all(unpack(x_packed)[0] == x)