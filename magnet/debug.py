def overfit(trainer, batch_size, epochs=1, metric='loss', sample_space=None, ax=None):
    from matplotlib import pyplot as plt
    if sample_space is None:
        _, ax = plt.subplots()
        epochs *= 100
        for sample_space in (1, 2, 4, 8, 16):
            overfit(trainer, batch_size=1, epochs=epochs,
                    metric=metric, sample_space=sample_space, ax=ax)
        overfit(trainer, batch_size=16, epochs=epochs, metric=metric, sample_space=16, ax=ax)
        bs = min(batch_size, 16)
        if bs > 16:
            overfit(trainer, bs, epochs, metric, sample_space=16, ax=ax)
        sample_length = int(len(trainer.data['train']) * 0.01)
        bs = min(batch_size, sample_length)
        if sample_length > 16:
            overfit(trainer, bs, epochs, metric, sample_length, ax)

        plt.show()
        return

    data = trainer.data
    with trainer.mock():
        trainer.data = (data(batch_size, sample_space=sample_space),
                        data(mode='val', sample_space=1))
        trainer.train(epochs, monitor_freq=10 / epochs, save_interval='2 d')
        trainer.history.show(metric, x_key='epochs', xlabel='epochs',
                             validation=False, ax=ax, log=True,
                             label=f'{batch_size}, {sample_space}')

    trainer.data = data