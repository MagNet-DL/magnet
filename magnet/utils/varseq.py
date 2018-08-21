import torch, numpy as np, magnet as mag

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence

def pack(sequences, lengths=None):
    from types import MethodType

    n = len(sequences) if isinstance(sequences, (tuple, list)) else len(sequences[0])
    shape = sequences[0].shape[1:] if isinstance(sequences, (tuple, list)) else sequences.shape[2:]

    if lengths is None:
        lengths = list(map(len, sequences))
        order = np.argsort(lengths)[::-1]
        sequences = [sequences[i] for i in order]
        sequences = pack_sequence(sequences)
        sequences.order = order
    else:
        order = np.argsort(lengths)[::-1]
        sequences = sequences[:, order]
        lengths = lengths[order]
        sequences = pack_padded_sequence(sequences, torch.tensor(lengths))

    sequences.order = order
    sequences.unpack = MethodType(lambda self, as_list=False: unpack(self, as_list), sequences)
    sequences.shape = torch.Size([-1, n] + list(shape))
    return sequences

def unpack(sequence, as_list=False):
    order = sequence.order

    sequences, lengths = pad_packed_sequence(sequence)
    order = np.argsort(order)

    sequences = sequences[:, order]; lengths = lengths[order]
    if not as_list: return sequences, lengths

    return [sequence[:l.item()] for sequence, l in zip(sequences.transpose(0, 1), lengths)]

def sort(sequences, order, dim=0):
    return torch.index_select(sequences, dim, torch.tensor(order.copy(), device=sequences.device))

def unsort(sequences, order, dim=0):
    return torch.index_select(sequences, dim, torch.tensor(np.argsort(order.copy()), device=sequences.device))