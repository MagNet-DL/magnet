from magnet.utils.plot import smooth_plot

class History(dict):
    r"""A dictionary-like repository which is used to store several metrics of
    interest in training in the form of snapshots.

    This object can be utilized to collect, store and analyze training metrics
    against a variety of features of interest (epochs, iterations, time etc.)

    Since this is a subclass of ``dict``, it can be used as such. However, it is
    prefered to operate it using the class-specific methods.

    Examples::

        >>> history = History()

        >>> # Add a simple value with a time stamp.
        >>> # This is like the statement: history['loss'] = 69
        >>> # However, any additional stamps can also be attached.
        >>> history.append('loss', 69, time=time())
        {'loss': [{'val': 69, 'time': 1535095251.6717412}]}

        >>> history.clear()

        >>> # Use a small buffer-size of 10.
        >>> # This means that only the latest 10 values are kept.
        >>> for i in range(100): history.append('loss', i, buffer_size=10)

        >>> # Flush the buffer with a time stamp.
        >>> history.flush(time=time())

        >>> # The mean of the last 10 values is now stored.
        {'loss': [{'val': 94.5, 'time': 1535095320.9745226}]}

    """
    def find(self, key):
        r"""A helper method that returns a filtered dictionary
        with a search key.

        Args:
            key (str): The filter key

        Examples::

            >>> # Assume the history is empty with keys: ['loss', 'val_loss',
            >>> # 'encoder_loss', 'accuracy', 'wierd-metric']

            >>> history.find('loss')
            {'loss': [], 'val_loss': [], 'encoder_loss': []}
        """
        return {k: self[k] for k in self.keys() if key in k}

    def append(self, key, value, validation=False, buffer_size=None, **stamps):
        r"""Append a new snapshot to the history.

        Args:
            key (str): The dictionary key / name of the object
            value (object): The actual object
            valdiation (bool): Whether this is a validation metric.
                Default: ``False``
            buffer_size (int or None): The size of the buffer of the key.
                Default: ``None``

        * :attr:`validation` is just a convinient key-modifier.
          It appends ``'val_'`` to the key.

        * :attr:`buffer_size` defines the size of the storage buffer for the
          specific :attr:`key`.

          The latest :attr:`buffer_size` snapshots are stored.

          If None, the :attr:`key` is stored as is.

        .. note::

            Any further keyword arguments define :attr:`stamps` that are
            essentially the signatures for the snapshot.
        """
        if validation: key = 'val_' + key

        try:
            self[key].append(value, buffer=(buffer_size is not None), **stamps)
        except KeyError:
            # If key does not exist, add it as new.
            self[key] = SnapShot(buffer_size)
            self[key].append(value, buffer=(buffer_size is not None), **stamps)

    def show(self, key=None, log=False, x_key=None, validation=True, legend=None, **kwargs):
        r""" Plot the snapshots for a key against a stamp.

        Args:
            key (str): The key of the record
            log (bool): If ``True``, the y-axis will be log-scaled.
                Default: ``False``
            x_key (str or None): The stamp to use as the x-axis.
                Default: ``None``
            validation (bool): Whether to plot the validation records
                (if they exist) as well. Default: ``True``
            legend (str or None): The legend entry. Default: ``None``

        Keyword Args:
            ax (pyplot axes object): The axis to plot into. Default: ``None``
            smoothen (bool): If ``True``, smoothens the plot. Default: ``True``
            window_fraction (float): How much of the plot to use as a window
                for smoothing. Default: :math:`0.3`
            gain (float): How much more dense to make the plot.
                Default: :math:`10`
            replace_outliers (bool): If ``True``, replaces outlier datapoints
                by a sensible value. Default: ``True``

        * :attr:`key` can be ``None``, in which case this method is successively
          called for all existing keys.
          The :attr:`log` attribute is overriden, however.
          It is only set to ``True`` for any key with ``'loss'`` in it.

        * :attr:`legend` can be ``None``, in which case the default legends
          ``'training'`` and ``'validation'`` are applied respectively.
        """
        from matplotlib import pyplot as plt

        ax = kwargs.pop('ax', None)

        if key is None:
            for k in self.keys():
                if 'val_' in k: continue
                self.show(k, 'loss' in k, x_key, validation, **kwargs)
                plt.show()
            return

        if ax is None: fig, ax = plt.subplots()
        label = 'training' if legend is None else legend
        self[key].show(ax, x_key, label=label, **kwargs)

        if validation:
            try:
                label = 'validation' if legend is None else legend
                self['val_' + key].show(ax, x_key, label=label)
            except KeyError: pass

        if log: plt.yscale('log')

        plt.ylabel(key.title())
        if isinstance(x_key, str):
            plt.xlabel(x_key)
            plt.title(f'{key.title()} vs {x_key.title()}')
        elif isinstance(x_key, str):
            plt.xlabel(x_key)
            plt.title(f'{key.title()} vs {x_key.title()}')
        else: plt.title(key.title())

        plt.legend()

    def flush(self, key=None, **stamps):
        r""" Flush the buffer (if exists) and append the mean.

        Args:
            key (str or None): The key to flush. Default: ``None``

        * :attr:`key` can be None, in which case this method is successively
          called for all existing keys.

        .. note::
            Any further keyword arguments define :attr:`stamps` that are
            essentially the signatures for the snapshot.
        """
        if key is None:
            for k in self.keys(): self.flush(k, **stamps)
            return

        self[key].flush(**stamps)

class SnapShot:
    r""" A list of stamped values (snapshots).

    This is used by the History object to store
    a repository of training metrics.

    Args:
        buffer_size (int): The size of the buffer. Default: :math:`-1`

    * If :attr:`buffer_size` is negative, then the snapshots are stored as is.
    """
    def __init__(self, buffer_size=-1):
        self._snaps = []
        if buffer_size is not None:
            self._buffer_size = buffer_size
            self._buffer = SnapShot(buffer_size=None)

    def append(self, value, buffer=False, **stamps):
        r""" Add a new snapshot.

        Args:
            value (object): The value to add
            buffer (bool): If ``True``, adds to the buffer instead.
                Default: ``False``

        .. note::

            Any further keyword arguments define :attr:`stamps` that are
            essentially the signatures for the snapshot.
        """
        if buffer:
            self._buffer.append(value, **stamps)

            # Remove the first value if buffer overflowed.
            if self._buffer_size > 0 and len(self._buffer) > self._buffer_size: self._buffer._pop(0)
            return

        self._snaps.append(dict(val=value, **stamps))

    def flush(self, **stamps):
        r""" Flush the buffer (if exists) and append the mean.

        .. note::

            Any keyword arguments define :attr:`stamps` that are
            essentially the signatures for the snapshot.
        """
        if not hasattr(self, '_buffer') or len(self._buffer) == 0: return

        values = self._buffer._retrieve()
        value = sum(values) / len(values)

        self.append(value, **stamps)

        # Clear the entire buffer if the buffer size is not finite.
        if self._buffer_size < 0: self._buffer._clear()

    def _retrieve(self, key='val', stamp=None):
        if stamp is None: return [snap[key] for snap in self._snaps]
        return list(zip(*[(snap[stamp], snap[key]) for snap in self._snaps if stamp in snap.keys()]))

    def _pop(self, index):
        self._snaps.pop(index)

    def _clear(self):
        self._snaps = []

    def __len__(self):
        return len(self._snaps)

    def __getitem__(self, index):
        return self._snaps[index]['val']

    def __repr__(self):
        return repr(self._snaps)

    def show(self, ax, x=None, label=None, **kwargs):
        r""" Plot the snapshots against a stamp.

        Args:
            ax (pyplot axes object): The axis to plot into
            x (str or None): The stamp to use as the x-axis. Default: ``None``
            label (str or None): The label for the line. Default: ``None``

        * :attr:`key` can be None, in which case this method is successively
          called for all existing keys.
          The :attr:`log` attribute is overriden, however.
          It is only set to ``True`` for any key with ``'loss'`` in it.

        * :attr:`legend` can be ``None``, in which case the default legends
          ``'training'`` and ``'validation'`` are applied respectively.

        Keyword Args:
             : See :py:meth:`History.show` for more details.

        .. note::
            Any further keyword arguments are passed to the plot function.
        """
        if x is None:
            x = list(range(len(self)))
            y = self._retrieve()
        else:
            x, y = self._retrieve(stamp=x)

        if len(x) != 0:
            window_fraction = kwargs.pop('window_fraction', 0.3)
            gain = kwargs.pop('gain', 10)
            replace_outliers = kwargs.pop('replace_outliers', True)

            if kwargs.pop('smoothen', True):
                line, = ax.plot(x, y, alpha=0.3)
                smooth_plot(x, y, label=label, ax=ax, c=line.get_color(),
                            window_fraction=window_fraction, gain=gain,
                            replace_outliers=replace_outliers, **kwargs)
            else:
                ax.plot(x, y, label=label, **kwargs)