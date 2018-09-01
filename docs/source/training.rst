magnet.training
===================================

.. automodule:: magnet.training

Trainer
^^^^^^^^^^^^^^^^^

.. autoclass:: Trainer
   :members: optimize, train, mock, epochs, register_parameter
   :member-order: bysource

SupervisedTrainer
^^^^^^^^^^^^^^^^^

.. autoclass:: SupervisedTrainer

.. autofunction:: finish_training

magnet.training.callbacks
===================================

.. automodule:: magnet.training.callbacks

CallbackQueue
^^^^^^^^^^^^^^^^^

.. autoclass:: CallbackQueue	
   :members: __call__, find

Monitor
^^^^^^^^^^^^^^^^^

.. autoclass:: Monitor
   :members: __call__, show

Validate
^^^^^^^^^^^^^^^^^

.. autoclass:: Validate
   :members: __call__

Checkpoint
^^^^^^^^^^^^^^^^^

.. autoclass:: Checkpoint
   :members: __call__

ColdStart
^^^^^^^^^^^^^^^^^

.. autoclass:: ColdStart
   :members: __call__

LRScheduler
^^^^^^^^^^^^^^^^^

.. autoclass:: LRScheduler
   :members: __call__

magnet.training.history
===================================

.. automodule:: magnet.training.history

.. autoclass:: History
   :members: append, flush, show, find
   :member-order: bysource

.. autoclass:: SnapShot
   :members: append, flush, show
   :member-order: bysource

magnet.training.utils
===================================

.. automodule:: magnet.training.utils
   :members:







