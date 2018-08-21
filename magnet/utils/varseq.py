import torch, numpy as np, magnet as mag

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence

def pack(sequences, lengths=None):
    if lengths is None:
        lengths = list(map(len, sequences))
        order = np.argsort(lengths)[::-1]
        sequences = [sequences[i] for i in order]
        return pack_sequence(sequences), order

    order = np.argsort(lengths)[::-1]
    sequences = sequences[:, order]
    lengths = lengths[order]

    return pack_padded_sequence(sequences, torch.tensor(lengths)), order

def unpack(sequence, order):
    sequences, lengths = pad_packed_sequence(sequence)
    order = np.argsort(order)

    return sequences[:, order], lengths[order]