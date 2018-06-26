class Trainer:
    def train(self, iterations=1):
        self.on_training_start()
        for batch in range(iterations):
            try:
                if not batch % self.batches_per_epoch:
                    self.on_epoch_start(int(batch // self.batches_per_epoch))
            except AttributeError: pass
            self.in_batch(batch)
            try:
                if not (batch + 1) % self.batches_per_epoch:
                    self.on_epoch_end(int(batch // self.batches_per_epoch))
            except AttributeError: pass
        self.on_training_end()

    def on_training_start(self):
        pass

    def on_epoch_start(self, epoch):
        pass

    def in_batch(self, batch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_training_end(self):
        pass

    def __call__(self, iterations=1):
        return self.train(iterations)

class SimpleTrainer(Trainer):
    def __init__(self, model, data, loss, optimizer='adam'):
        self.model = model
        self.data = data
        self.loss = loss
        self.optimizer = self._get_optimizer(optimizer)

    def on_training_start(self):
        self.dl = iter(self.data())
        self.batches_per_epoch = len(self.dl)

    def in_batch(self, batch):
        x, y = next(self.dl)
        l = self.loss(self.model(x), y)
        l.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _get_optimizer(self, optimizer):
        from torch import optim

        if optimizer == 'adam':
            return optim.Adam(self.model.parameters(), amsgrad=True)