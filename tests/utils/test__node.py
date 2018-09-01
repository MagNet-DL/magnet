import torch

from torch import nn

import magnet.nodes as mn
from magnet.utils import summarize

def test_summarize_node():
    model = mn.Linear(10, act=None)
    summarize(model, torch.randn(1, 1, 28, 28), arguments=True)

def test_summarize_module():
    model = nn.Linear(784, 10)
    summarize(model, torch.randn(1, 784), arguments=True)