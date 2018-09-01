# coding=utf-8
import torch.nn.functional as F

from torch import nn

from .nodes import Node

class Lambda(Node):
    r"""Wraps a Node around any function.

    Args:
        fn (callable): The function which gets called in the forward pass

    Examples::

        >>> import magnet.nodes as mn

        >>> import torch

        >>> model = mn.Lambda(lambda x: x.mean())

        >>> model(torch.arange(5, dtype=torch.float)).item()
        2.0

        >>> def subtract(x, y):
        >>>     return x - y

        >>> model = mn.Lambda(subtract)

        >>> model(2 * torch.ones(1), torch.ones(1)).item()
        1.0
    """

    def __init__(self, fn, **kwargs):
        super().__init__(fn, **kwargs)

        # If a name is not supplied, get the function name instead
        # of the class (Lambda) name.
        if self.name == self.__class__.__name__:
            self.name = self._args['fn'].__name__

    def forward(self, *args, **kwargs):
        return self._args['fn'](*args, **kwargs)

class Conv(Node):
    r"""Applies a convolution over an input tensor.

    Args:
        c (int): Number of channels produced by the convolution.
            Default: Inferred
        k (int or tuple): Size of the convolving kernel. Default: ``3``
        p (int, tuple or str): Zero-padding added to both sides
            of the input. Default: ``'half'``
        s (int or tuple): Stride of the convolution. Default: ``1``
        d (int or tuple): Spacing between kernel elements. Default: ``1``
        g (int): Number of blocked connections from input channels
            to output channels. Default: ``1``
        b (bool): If ``True``, adds a learnable bias to the output.
            Default: ``True``
        ic (int): Number of channels in the input image.
            Default: Inferred
        act (str or None): The activation function to use.
            Default: ``'relu'``

    * :attr:`p` can be conveniently used for ``'half'``, ``'same'`` or
      ``'double'`` padding to half, same or double the image size respectively.
      The arguments are accordingly inferred at runtime.
      For ``'half'`` padding, the output channels (if not provided)
      are set to twice the input channels to make up for the lost
      information and vice-versa for the double padding.
      For ``'same'`` padding, the output channels are kept equal to the
      input channels.
      In all three cases, the dilation is set to ``1`` and the stride
      is modified as required.

    * :attr:`c` is inferred from the second dimension of the
      input tensor.

    * :attr:`act` is set to ``'relu'`` by default unlike the PyTorch
      implementation where activation functions need to be seperately
      defined.
      Take caution to manually set the activation to ``None``, where needed.

    .. note::

         The dimensions (1, 2 or 3) of the convolutional kernels
         are inferred from the corresponding shape of the input tensor.

    .. note::

         One can also create multiple Nodes using the convinient
         multiplication (``*``) operation.

         Multiplication with an integer :math:`n`, gives :math:`n`
         copies of the Node.

         Multiplication with a list or tuple of integers,
         :math:`(c_1, c_2, ..., c_n)` gives :math:`n` copies
         of the Node with :attr:`c` set to :math:`c_i`

    Shape:
    - Input: :math:`(N, C_{in}, *)`
    where `*` is any non-zero number of trailing dimensions.
    - Output: :math:`(N, C_{out}, *)`

    Attributes:
        layer (nn.Module): The Conv module built from torch.nn

    Examples::

        >>> import torch

        >>> from torch import nn

        >>> import magnet.nodes as mn
        >>> from magnet.utils import summarize

        >>> # A Conv layer with 32 channels and half padding
        >>> model = mn.Conv(32)

        >>> model(torch.randn(4, 16, 28, 28)).shape
        torch.Size([4, 32, 14, 14])

        >>> # Alternatively, the 32 in the constructor may be omitted
        >>> # since it is inferred on runtime.

        >>> # The same conv layer with 'double' padding
        >>> model = mn.Conv(p='double')

        >>> model(torch.randn(4, 16, 28, 28)).shape
        torch.Size([4, 8, 56, 56])

        >>> layers = mn.Conv() * 3
        [Conv(), Conv(), Conv()]

        >>> model = nn.Sequential(*layers)
        >>> summarize(model)
        +-------+------------+----------------------+
        | Node  |   Shape    | Trainable Parameters |
        +-------+------------+----------------------+
        | input | 16, 28, 28 |          0           |
        +-------+------------+----------------------+
        | Conv  | 32, 14, 14 |        4,640         |
        +-------+------------+----------------------+
        | Conv  |  64, 7, 7  |        18,496        |
        +-------+------------+----------------------+
        | Conv  | 128, 4, 4  |        73,856        |
        +-------+------------+----------------------+
        Total Trainable Parameters: 96,992
    """
    def __init__(self, c=None, k=3, p='half', s=1, d=1, g=1, b=True, ic=None, act='relu', bn=False, **kwargs):
        super().__init__(c, k, p, s, d, g, b, ic, act, bn, **kwargs)

    def build(self, x):
        from magnet.nodes.functional import wiki

        self._set_padding(x) # Handle 'half', 'same' and 'double' padding

        # Infer the input shape if not given
        if self._args['ic'] is None: self._args['ic'] = x.shape[1]

        self._activation = wiki['activations'][self._args['act']]

        layer_class = self._find_layer(x) # Infer the layer (Conv1D, 2D or 3D)

        self.layer = layer_class(kernel_size=self._args['k'], out_channels=self._args['c'],
                                stride=self._args['s'], padding=self._args['p'], dilation=self._args['d'],
                                groups=self._args['g'], bias=self._args['b'], in_channels=self._args['ic'])

        if self._args['bn']: self._batch_norm = BatchNorm()
        super().build(x)

    def forward(self, x):
        if hasattr(self, '_upsample'): x = F.interpolate(x, scale_factor=self._upsample)

        x = self._activation(self.layer(x))

        if self._args['bn']: x = self._batch_norm(x)

        return x

    @staticmethod
    def _find_layer(x):
        shape_dict = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        ndim = len(x.shape) - 2
        return shape_dict[ndim - 1]

    def _set_padding(self, x):
        in_shape = x.shape

        p = self._args['p']

        if p == 'half': f = 0.5
        elif p == 'same': f = 1
        elif p == 'double':
            self._upsample = 2
            if self._args['c'] is None:
                self._args['c'] = in_shape[1] // 2
            f = 1
        else: return

        s = 1 / f

        self._args['d'] = 1
        self._args['s'] = int(s)
        self._args['p'] = int(self._args['k'] // 2)
        if self._args['c'] is None:
            self._args['c'] = self._args['s'] * in_shape[1]

    def _mul_list(self, n):
        convs = [self]
        self._args['c'] = n[0]
        kwargs = self._args.copy()
        for c in n[1:]:
            kwargs['c'] = c
            convs.append(self.__class__(**kwargs))

        return convs

class Linear(Node):
    r"""Applies a linear transformation to the incoming tensor

    Args:
        o (int or tuple): Output dimensions. Default: :math:`1`
        b (bool): Whether to include a bias term. Default: ``True``
        flat (bool): Whether to flatten out the input to 2 dimensions.
            Default: ``True``
        i (int): Input dimensions. Default: Inferred
        act (str or None): The activation function to use.
            Default: ``'relu'``
        bn (bool): Whether to use Batch Normalization immediately after
            the layer. Default: ``False``

    * :attr:`flat` is used by default to flatten the input to a vector.
      This is useful, say in the case of CNNs where an 3-D image based
      output with multiple channels needs to be fed to several dense layers.

    * :attr:`o` is inferred from the last dimension of the
      input tensor.

    * :attr:`act` is set to 'relu' by default unlike the PyTorch
      implementation where activation functions need to be seperately
      defined.
      Take caution to manually set the activation to None, where needed.

    .. note::

        One can also create multiple Nodes using the convinient
        multiplication (*) operation.

        Multiplication with an integer :math:`n`, gives :math:`n`
        copies of the Node.

        Multiplication with a list or tuple of integers,
        :math:`(o_1, o_2, ..., o_n)` gives :math:`n` copies
        of the Node with :attr:`o` set to :math:`o_i`

    .. note::

        If :attr:`o` is a tuple, the output features are its product
        and the output is inflated to this shape.

    Shape:
        If :attr:`flat` is True
            - Input: :math:`(N, *)` where :math:`*` means any number of
              trailing dimensions
            - Output: :math:`(N, *)`
        Else
            - Input: :math:`(N, *, in\_features)` where :math:`*` means any
              number of trailing dimensions
            - Output: :math:`(N, *, out\_features)` where all but the last
              dimension are the same shape as the input.

    Attributes:
        layer (nn.Module): The Linear module built from torch.nn

    Examples::

        >>> import torch

        >>> from torch import nn

        >>> import magnet.nodes as mn
        >>> from magnet.utils import summarize

        >>> # A Linear mapping to 10-dimensional space
        >>> model = mn.Linear(10)

        >>> model(torch.randn(64, 3, 28, 28)).shape
        torch.Size([64, 10])

        >>> # Don't flatten the input
        >>> model = mn.Linear(10, flat=False)

        >>> model(torch.randn(64, 3, 28, 28)).shape
        torch.Size([64, 3, 28, 10])

        >>> # Make a Deep Neural Network
        >>> # Don't forget to turn the activation to None in the final layer
        >>> layers = mn.Linear() * (10, 50) + [mn.Linear(10, act=None)]
        [Linear(), Linear(), Linear()]

        >>> model = nn.Sequential(*layers)
        >>> summarize(model)
        +------+---------+--------------------+----------------------------------------------------+
        | Node |  Shape  |Trainable Parameters|                   Arguments                        |
        +------+---------+--------------------+----------------------------------------------------+
        |input |3, 28, 28|         0          |                                                    |
        +------+---------+--------------------+----------------------------------------------------+
        |Linear|   10    |       23,530       |bn=False, act=relu, i=2352, flat=True, b=True, o=10 |
        +------+---------+--------------------+----------------------------------------------------+
        |Linear|   50    |        550         |bn=False, act=relu, i=10, flat=True, b=True, o=50   |
        +------+---------+--------------------+----------------------------------------------------+
        |Linear|   10    |        510         |bn=False, act=None, i=50, flat=True, b=True, o=10   |
        +------+---------+--------------------+----------------------------------------------------+
        Total Trainable Parameters: 24,590
    """
    def __init__(self, o=1, b=True, flat=True, i=None, act='relu', bn=False, **kwargs):
        super().__init__(o, b, flat, i, act, bn, **kwargs)

    def build(self, x):
        from numpy import prod
        from magnet.nodes.functional import wiki

        # Infer the input shape if not given
        if self._args['i'] is None: self._args['i'] = prod(x.shape[1:]) if self._args['flat'] else x.shape[-1]

        # If a tuple is given as output shape, inflate to that tuple
        if isinstance(self._args['o'], (list, tuple)):
            self._inflate_shape = self._args['o']
            self._args['o'] = prod(self._args['o'])
        else:
            self._inflate_shape = None

        self._activation = wiki['activations'][self._args['act']]

        self.layer = nn.Linear(*[self._args[k] for k in ('i', 'o', 'b')])

        if self._args['bn']: self._batch_norm = BatchNorm()

        super().build(x)

    def forward(self, x):
        if self._args['flat']: x = x.view(x.size(0), -1)

        x = self._activation(self.layer(x))

        if self._args['bn']: x = self._batch_norm(x)

        if self._inflate_shape is not None: x = x.view(-1, *self._inflate_shape)

        return x

    def _mul_list(self, n):
        lins = [self]
        self._args['o'] = n[0]
        kwargs = self._args.copy()
        for o in n[1:]:
            kwargs['o'] = o
            lins.append(self.__class__(**kwargs))

        return lins

class _RNNBase(Node):
    def __init__(self, mode, h, n=1, b=False, bi=False, act='tanh', d=0, batch_first=False, i=None, **kwargs):
        self.layer = mode
        super().__init__(h, n, b, bi, act, d, batch_first, i, **kwargs)

    def build(self, x, h=None):
        # Infer the input shape if not given
        if self._args['i'] is None: self._args['i'] =  x.shape[-1]

        self.layer = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[self.layer.lower()]

        kwargs = {'nonlinearity': self._args['act'], 'bias': self._args['b'],
                'batch_first': self._args['batch_first'],
                'dropout': self._args['d'], 'bidirectional': self._args['bi']}

        # The 'nonlinearity' / 'act' argument is not a part of LSTM and GRU
        if not isinstance(self.layer, nn.RNN): kwargs.pop('nonlinearity')

        self.layer = self.layer(*[self._args[k] for k in ('i', 'h', 'n')], **kwargs)

        super().build(x, h)

    def forward(self, x, h=None):
        return self.layer(x, h)

    def _mul_list(self, n):
        rnns = [self]
        self._args['h'] = n[0]
        kwargs = self._args.copy()
        for h in n[1:]:
            kwargs['h'] = h
            print(self.__class__, kwargs)
            rnns.append(self.__class__(**kwargs))

        return rnns

class RNN(_RNNBase):
    r"""Applies a multi-layer RNN with to an input tensor.

    Args:
        h (int, Required): The number of features in the hidden state `h`
        n (int):  Number of layers. Default: ``1``
        b (bool): Whether to include a bias term. Default: ``True``
        bi (bool): If ``True``, becomes a bidirectional RNN.
            Default: ``False``
        act (str or None): The activation function to use.
            Default: ``'tanh'``
        d (int): The dropout probability of the outputs of each layer.
            Default: ``0``
        batch_first (False): If ``True``, then the input and output
            tensors are provided as ``(batch, seq, feature)``.
            Default: ``False``
        i (int): Input dimensions. Default: Inferred

    * :attr:`i` is inferred from the last dimension of the
      input tensor.

    .. note::

         One can also create multiple Nodes using the convinient
         multiplication (*) operation.

         Multiplication with an integer :math:`n`, gives :math:`n`
         copies of the Node.

         Multiplication with a list or tuple of integers,
         :math:`(h_1, h_2, ..., h_n)` gives :math:`n` copies
         of the Node with :attr:`h` set to :math:`h_i`

    Attributes:
        layer (nn.Module): The RNN module built from torch.nn

    Examples::

        >>> import torch

        >>> from torch import nn

        >>> import magnet.nodes as mn
        >>> from magnet.utils import summarize

        >>> # A recurrent layer with 32 hidden dimensions
        >>> model = mn.RNN(32)

        >>> model(torch.randn(7, 4, 300))[0].shape
        torch.Size([7, 4, 32])

        >>> # Attach a linear head
        >>> model = nn.Sequential(model, mn.Linear(1000, act=None))
    """
    def __init__(self, h, n=1, b=False, bi=False, act='tanh', d=0, batch_first=False, i=None, **kwargs):
        mode = kwargs.pop('mode', 'rnn')
        super().__init__(mode, h, n, b, bi, act, d, batch_first, i, **kwargs)

class LSTM(_RNNBase):
    r"""Applies a multi-layer LSTM with to an input tensor.

            See mn.RNN for more details
            """
    def __init__(self, h, n=1, b=False, bi=False, d=0, batch_first=False, i=None, **kwargs):
        act = kwargs.pop('act', None)
        mode = kwargs.pop('mode', 'lstm')
        super().__init__(mode, h, n, b, bi, act, d, batch_first, i, **kwargs)

class GRU(_RNNBase):
    r"""Applies a multi-layer GRU with to an input tensor.

    See mn.RNN for more details
    """
    def __init__(self, h, n=1, b=False, bi=False, d=0, batch_first=False, i=None, **kwargs):
        act = kwargs.pop('act', None)
        mode = kwargs.pop('mode', 'gru')
        super().__init__(mode, h, n, b, bi, act, d, batch_first, i, **kwargs)

class BatchNorm(Node):
    r"""Applies Batch Normalization to the input tensor
    e=1e-05, m=0.1, a=True, track=True, i=None

    Args:
        e (float): A small value added to the denominator
            for numerical stability. Default: ``1e-5``
        m (float or None): The value used for the running_mean
            and running_var computation. Can be set to ``None`` for
            cumulative moving average (i.e. simple average). Default: ``0.1``
        a (bool): Whether to have learnable affine parameters.
            Default: ``True``
        track (bool): Whether to track the running mean and variance.
            Default: ``True``
        i (int): Input channels. Default: Inferred

    * :attr:`i` is inferred from the second dimension of the
      input tensor.

    .. note::

         The dimensions (1, 2 or 3) of the running mean and variance
         are inferred from the corresponding shape of the input tensor.

    .. note::

         One can also create multiple Nodes using the convinient
         multiplication (*) operation.

         Multiplication with an integer :math:`n`, gives :math:`n`
         copies of the Node.

         Multiplication with a list or tuple of integers,
         :math:`(i_1, i_2, ..., i_n)` gives :math:`n` copies
         of the Node with :attr:`i` set to :math:`i_i`

    Shape:
        - Input: :math:`(N, C, *)` where :math:`*` means any number of
          trailing dimensions
        - Output: :math:`(N, C, *)` (same shape as input)

    Attributes:
        layer (nn.Module): The BatchNorm module built from :py:class:`torch.nn`

    Examples::

        >>> import torch

        >>> from torch import nn

        >>> import magnet.nodes as mn
        >>> from magnet.utils import summarize

        >>> # A Linear mapping to 10-dimensional space
        >>> model = mn.Linear(10)

        >>> model(torch.randn(64, 3, 28, 28)).shape
        torch.Size([64, 10])

        >>> # Don't flatten the input
        >>> model = mn.Linear(10, flat=False)

        >>> model(torch.randn(64, 3, 28, 28)).shape
        torch.Size([64, 3, 28, 10])

        >>> # Make a Deep Neural Network
        >>> # Don't forget to turn the activation to None in the final layer
        >>> layers = mn.Linear() * (10, 50) + [mn.Linear(10, act=None)]
        [Linear(), Linear(), Linear()]

        >>> model = nn.Sequential(*layers)
        >>> summarize(model)
        +------+---------+--------------------+----------------------------------------------------+
        | Node |  Shape  |Trainable Parameters|                   Arguments                        |
        +------+---------+--------------------+----------------------------------------------------+
        |input |3, 28, 28|         0          |                                                    |
        +------+---------+--------------------+----------------------------------------------------+
        |Linear|   10    |       23,530       |bn=False, act=relu, i=2352, flat=True, b=True, o=10 |
        +------+---------+--------------------+----------------------------------------------------+
        |Linear|   50    |        550         |bn=False, act=relu, i=10, flat=True, b=True, o=50   |
        +------+---------+--------------------+----------------------------------------------------+
        |Linear|   10    |        510         |bn=False, act=None, i=50, flat=True, b=True, o=10   |
        +------+---------+--------------------+----------------------------------------------------+
        Total Trainable Parameters: 24,590
    """
    def __init__(self, e=1e-05, m=0.1, a=True, track=True, i=None, **kwargs):
        super().__init__(e, m, a, track, i, **kwargs)

    def build(self, x):
        # Infer the input shape if not given
        self._args['i'] = x.shape[1]

        layer_class = self._find_layer(x) # Infer the layer (BatchNorm1D, 2D or 3D)
        self.layer = layer_class(*[self._args[k] for k in ('i', 'e', 'm', 'a', 'track')])

        super().build(x)

    def forward(self, x):
        return self.layer(x)

    def _find_layer(self, x):
        shape_dict = [nn.BatchNorm1d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        ndim = len(x.shape) - 1
        return shape_dict[ndim - 1]
