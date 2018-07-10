def caller_locals(ancestor=False):
    """Print the local variables in the caller's frame."""
    import inspect
    frame = inspect.currentframe().f_back.f_back

    try:
        l = frame.f_locals

        if ancestor:
            f_class = l.pop('__class__', None)
            caller = l.pop('self')
            while f_class is not None and isinstance(caller, f_class):
                l = frame.f_locals
                l.pop('self', None)
                frame = frame.f_back
                f_class = frame.f_locals.pop('__class__', None)

        l.pop('self', None)
        l.pop('__class__', None)
        return l
    finally: del frame

def num_params(module):
    from numpy import prod

    trainable, non_trainable = 0, 0
    for p in module.parameters():
        n = prod(p.size())
        if p.requires_grad:
            trainable += n
        else:
            non_trainable += n

    return trainable, non_trainable

def summarize(self, module, parameters='trainable', arguments=False, batch=False, children=False, max_width=120):
        if children:
            for c in module.children(): c.summarize(parameters, arguments, batch, children=False, max_width=max_width)

        def _summary_table_helper(mode, parameters='trainable', node=None):
            str_dict = {'trainable': 'Trainable', 'non-trainable': 'NON-Trainable', 'all': '', True: '(Trainable, NON-Trainable)'}
            if mode == 'col': return str_dict[parameters] + ' Parameters'

            def _get_num_params(module):
                n = num_params(module) if module is not None else (0, 0)
                n_dict = {'trainable': n[0], 'non-trainable': n[1], 'all': sum(n), True: n}
                n = n_dict[parameters]
                return ', '.join(['{:,}'] * len(n)).format(*n) if type(n) is tuple else '{:,}'.format(n)

            if mode == 'row': return _get_num_params(node)

            print('Total ' + str_dict[parameters] + ' Parameters:', _get_num_params(node))

        header = ['Name', 'Shape']

        shapes = None
        if len(shapes) == 1: shapes = shapes[0]
        row = [self.name, shapes]
        if parameters is not False:
            header.append(_summary_table_helper('col', parameters))
            row.append(_summary_table_helper('row', parameters, module))
        if arguments:
            header.append('Arguments')
            if isinstance(module, Node): row.append(module.get_args())
            else: row.append('')

        from beautifultable import BeautifulTable
        table = BeautifulTable(max_width=max_width)
        table.column_headers = header
        input_row = ['Input', ]
        table.append_row(row)
        print(table)

def get_output_shape(module, input_shape):
    from torch import no_grad, randn

    with no_grad(): return tuple(module(randn(input_shape)).size())

def to_node(x, input_shape=None):
        from .nodes import Lambda
        from .models import Sequential
        from inspect import isfunction

        if type(x) is dict:
            name = list(x.keys())[0]

            node = to_node(x[name], input_shape)
            node.name = name
        elif type(x) in [list, tuple]: node = Sequential(*x, x=input_shape)
        elif isfunction(x): node = Lambda(x)
        else: node = x

        if not hasattr(node, 'name'): node.name = node.__class__.__name__

        return node

def get_function_name(fn):
    from inspect import getsource
    src = getsource(fn)

    name = src.split('=')
    if len(name) > 1: return name[0].strip()

    name = src.split('def ')
    if len(name) > 1: return name[1].split('(')[0].strip()

def get_tqdm():
    """
    :return: Returns a flexible tqdm object according to the environment of execution.
    """
    import tqdm

    try:
        get_ipython()
        return getattr(tqdm, 'tqdm_notebook')
    except:
        return getattr(tqdm, 'tqdm')
