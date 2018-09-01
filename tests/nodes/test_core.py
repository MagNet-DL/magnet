import torch

from torch import nn

import magnet as mag
from magnet.nodes.core import Lambda, Conv, Linear, RNN, LSTM, GRU, BatchNorm

torch.no_grad()

class TestLambda:
    def test_lambda_function_has_name_lambda(self):
        fn = lambda x: x ** 2
        node = Lambda(fn)
        assert node.name == '<lambda>'

    def test_function_name_transferred(self):
        def fn(x):
            return x**2

        node = Lambda(fn)
        assert node.name == 'fn'

    def test_square(self):
        x = torch.ones(4, 1, 28, 28, device=mag.device)

        node = Lambda(lambda x: x ** 2).eval()

        assert torch.all(node(x) == x ** 2)

class TestConv:
    def test_half_padding_args(self):
        x = torch.ones(4, 1, 28, 28)

        conv_half = Conv().eval()
        conv_half(x)

        conv = Conv(2, p=1, s=2).eval()
        conv(x)

        assert conv._args == conv_half._args

    def test_half_padding(self):
        conv = Conv().eval()
        assert conv(torch.ones(4, 1, 28, 28)).shape == (4, 2, 14, 14)

    def test_same_padding(self):
        conv = Conv(p='same').eval()
        assert conv(torch.ones(4, 1, 28, 28)).shape == (4, 1, 28, 28)

    def test_double_padding(self):
        conv = Conv(p='double').eval()
        assert conv(torch.ones(4, 2, 28, 28)).shape == (4, 1, 56, 56)

    def test_conv_1d(self):
        conv = Conv().eval()
        conv(torch.randn(1, 2, 3))
        assert isinstance(conv.layer, nn.Conv1d)

    def test_conv_2d(self):
        conv = Conv().eval()
        conv(torch.randn(1, 2, 3, 4))
        assert isinstance(conv.layer, nn.Conv2d)

    def test_conv_3d(self):
        conv = Conv().eval()
        conv(torch.randn(1, 2, 3, 4, 5))
        assert isinstance(conv.layer, nn.Conv3d)

    def test_mul_list(self):
        conv_layer = Conv().eval()
        cs = (5, 3, 2)
        convs = conv_layer * cs

        assert convs[0] is conv_layer
        assert all(conv._args[k] == conv_layer._args[k]
                   for k in conv_layer._args.keys() if k != 'c'
                   for conv in convs)
        assert all(conv._args['c'] == c for conv, c in zip(convs, cs))

class TestLinear:
    def test_flatten(self):
        linear = Linear()
        y = linear(torch.ones(4, 1, 28, 28))
        assert linear.layer.weight.shape == (1, 784)
        assert y.shape == (4, 1)

    def test_inflate(self):
        linear = Linear((1, 28, 28))
        y = linear(torch.ones(4, 1))
        assert linear.layer.weight.shape == (784, 1)
        assert y.shape == (4, 1, 28, 28)

    def test_mul_list(self):
        lin_layer = Linear().eval()
        os = (50, 30, 20)
        lins = lin_layer * os

        assert lins[0] is lin_layer
        assert all(lin._args[k] == lin_layer._args[k]
                   for k in lin_layer._args.keys() if k != 'o'
                   for lin in lins)
        assert all(lin._args['o'] == o for lin, o in zip(lins, os))

class TestRNN:
    base = RNN

    def test_shape(self):
        node = self.base(300).eval()
        x, h = node(torch.ones(7, 4, 100))
        assert x.shape == (7, 4, 300)

        if not isinstance(h, tuple): h = (h, )
        for h_i in h: assert h_i.shape == (1, 4, 300)

    def test_mul_list(self):
        rnn_layer = self.base(300).eval()
        hs = (300, 500, 200)
        rnns = rnn_layer * hs

        assert rnns[0] is rnn_layer
        assert all(rnn._args[k] == rnn_layer._args[k]
                   for k in rnn_layer._args.keys() if k != 'h'
                   for rnn in rnns)
        assert all(rnn._args['h'] == h for rnn, h in zip(rnns, hs))

class TestLSTM(TestRNN):
    base = LSTM

class TestGRU(TestRNN):
    base = GRU

class TestBatchNorm:
    def test_bn_1d(self):
        bn = BatchNorm().eval()
        bn(torch.randn(4, 2))
        assert isinstance(bn.layer, nn.BatchNorm1d)
        bn(torch.randn(4, 2, 3))
        assert isinstance(bn.layer, nn.BatchNorm1d)

    def test_bn_2d(self):
        bn = BatchNorm().eval()
        bn(torch.randn(4, 2, 3, 4))
        assert isinstance(bn.layer, nn.BatchNorm2d)

    def test_bn_3d(self):
        bn = BatchNorm().eval()
        bn(torch.randn(4, 2, 3, 4, 5))
        assert isinstance(bn.layer, nn.BatchNorm3d)