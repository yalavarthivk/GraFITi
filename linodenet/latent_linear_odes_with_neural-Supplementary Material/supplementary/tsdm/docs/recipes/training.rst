Vanilla Training Loop
---------------------

Typical experimental flow, given a fixed hyperparameter combination

.. code-block:: python

    from tsdm.datasets import DATASETS
    from tsdm.encoders import ENCODERS
    from tsdm.models   import MODELS
    from tsdm.losses import LOSSES
    from tsdm.optimizers import OPTIMIZERS
    from tsdm.dataloaders import DATALOADERS
    from tsdm.trainers import TRAINERS
    from tsdm.metrics import METRICS
    from tsdm.loggers import LOGGERS

    # current HP configuration
    HP = json.read("ID-XXXXXXX.json")

    dataset_cls = DATASETS[HP['dataset']]
    dataset     = dataset(HP['dataset_cfg'])

    model_cls = MODELS[HP['model']]
    model     = model_cls(HP['model_cfg'] + dataset.info)

    encoder_cls = ENCODERS[HP['encoder']]
    encoder     = encoder_cls(HP['encoder_cfg'] + dataset.info + model.info)

    optimizer_cls = OPTIMIZERS[HP['optimizer']]
    optimizer     = optimizer_cls[HP['optimizer_cfg']]

    loss_cls = LOSSES(HP['loss'])
    loss     = loss_cls(HP['loss_cfg'])

    dataloader_cls = DATALOADERS[HP['dataloader']]
    dataloader     = dataloader_cls(HP['dataloader_cfg'])

    trainer_cls   = TRAINERS[HP['trainer']]
    trainer       = HP['trainer_cfg']

    metric_cls = METRICS[HP['metrics']]
    metrics    = [metric_cls(config) for config, metric_cls in zip(metric_cls, HP['metrics_cfg'])]

    logger_cls = LOGGERS[HP['logger']]
    logger     = logger_cls(HP['logger_cfg'], model, optimizer, loss, metrics, ...)

    for batch in dataloader:
        x, y = batch
        x  = encoder.encode(x)
        yhat = model(x)
        yhat = encoder.decode(yhat)
        r = loss(y, yhat)
        r.backward()
        optimizer.step()
        for metric in metrics:
            metric(y, encoder.decode(yhat))
        logger.log()

        if trainer.stopping_criteria(model, optimizer, dataloader, logger.history):
            break
    else:
        warning(F"No convergence in {dataloader.epochs} epochs!!")

    return results
