import pytest

from magnet.nodes import Node

class TestNode:
    def test_not_built(self):
        node = Node()
        assert not node._built

    def test_store_arguments(self):
        args = [4, 2, 3]
        name = 'some_node'
        kwargs = {'r': 2, 'k': 4, 'tf': 34}

        node = Node(*args, name=name,**kwargs)

        assert node._args == {'args': tuple(args), **kwargs}
        assert node.name == name

    def test_name_is_not_senseless(self):
        with pytest.raises(ValueError):
            Node(name=None)
            Node(name='')

    def test_mul_int(self):
        n = 5
        node = Node(5, 2, a=1)
        nodes = node * n

        assert nodes[0] is node
        assert all(nodes[i]._args == node._args for i in range(1, n))

    def test_print_args(self):
        node = Node(5, 2, a=1)
        assert node.get_args() == 'args=(5, 2), a=1'

    def test_cannot_mul_list(self):
        with pytest.raises(NotImplementedError):
            node = Node() * (4, 2)
