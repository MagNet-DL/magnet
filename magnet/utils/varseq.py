import torch, numpy as np

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence

def pack(sequences, lengths=None):
    r"""Packs a list of variable length Tensors

    Args:
        sequences (list or torch.Tensor): The list of Tensors to pack
        lengths (list): list of lengths of each tensor. Default: ``None``

    .. note::
        If :attr:`sequences` is a tensor, :attr:`lengths` needs to be provided.

    .. note::
        The packed sequence that is returned has a convinient :py:meth:`unpack`
        method as well as ``shape`` and ``order`` attributes.
        The ``order`` attribute stores the sorting order which should be used
        for unpacking.

    Shapes:
        :attr:`sequences` should be a list of Tensors of size L x *,
        where L is the length of a sequence and * is any number of trailing
        dimensions, including zero.
    """
    from types import MethodType

    n = len(sequences) if isinstance(sequences, (tuple, list)) else len(sequences[0])
    shape = sequences[0].shape[1:] if isinstance(sequences, (tuple, list)) else sequences.shape[2:]

    # Check if a batched Tensor is provided (lengths is None)
    # or an explicit list of Tensors
    if lengths is None:
        lengths = list(map(len, sequences))
        order = np.argsort(lengths)[::-1]
        sequences = [sequences[i] for i in order]
        sequences = pack_sequence(sequences)
        sequences.order = order
    else:
        order = np.argsort(lengths)[::-1].copy()
        sequences = sequences[:, order]
        lengths = lengths[order]
        sequences = pack_padded_sequence(sequences, torch.tensor(lengths))

    sequences.order = order
    sequences.unpack = MethodType(lambda self, as_list=False: unpack(self, as_list), sequences)
    sequences.shape = torch.Size([-1, n] + list(shape))
    return sequences

def unpack(sequence, as_list=False):
    r"""Unpacks a ``PackedSequence`` object.

    Args:
        sequence (``PackedSequence``): The tensor to unpack.
        as_list (bool): If ``True``, returns a list of tensors.
            Default: ``False``

    .. note::
        The sequence should have an ``order`` attribute
        that stores the sorting order.
    """
    order = sequence.order

    sequences, lengths = pad_packed_sequence(sequence)
    order = np.argsort(order)

    sequences = sequences[:, order]; lengths = lengths[order]
    if not as_list: return sequences, lengths

    return [sequence[:l.item()] for sequence, l in zip(sequences.transpose(0, 1), lengths)]

def sort(sequences, order, dim=0):
    r"""Sorts a tensor in a certain order along a certain dimension.

    Args:
        sequences (torch.Tensor): The tensor to sort
        order (numpy.ndarray): The sorting order
        dim (int): The dimension to sort. Default ``0``
    """
    return torch.index_select(sequences, dim, torch.tensor(order.copy(), device=sequences.device))

def unsort(sequences, order, dim=0):
    r"""Unsorts a tensor in a certain order along a certain dimension.

    Args:
        sequences (torch.Tensor): The tensor to unsort
        order (numpy.ndarray): The sorting order
        dim (int): The dimension to unsort. Default ``0``
    """
    return torch.index_select(sequences, dim, torch.tensor(np.argsort(order.copy()), device=sequences.device))