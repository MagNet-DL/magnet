import pytest
import hypothesis.strategies as st

from hypothesis import given

from magnet.nodes import Node

class TestNode:
    def test_not_built(self):
        node = Node()
        assert not node._built

    @given(st.lists(st.integers(), max_size=5), st.booleans(),
           st.dictionaries(st.text(max_size=5), st.integers(), max_size=5),
           st.data())
    def test_store_arguments(self, args, include_name, kwargs, name):
        if include_name:
            name = kwargs['name'] = name.draw(st.text(min_size=1, max_size=10))

        node = Node(*args, **kwargs)
        kwargs.pop('name', None)

        assert node._args == {'args': tuple(args), **kwargs}
        if include_name: assert node.name == name

    @given(st.one_of(st.none(), st.just('')))
    def test_name_is_not_senseless(self, name):
        with pytest.raises(ValueError):
            Node(name=name)

    @given(st.integers(1, 10))
    def test_mul_int(self, n):
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
