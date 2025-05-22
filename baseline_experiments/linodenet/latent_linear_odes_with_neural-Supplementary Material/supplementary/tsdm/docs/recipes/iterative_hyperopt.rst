.. toctree::


Iterative Hyperparameter Optimization
-------------------------------------

For example: Gradient Based Hyperparameter Optimization.

Since gradient based methods are very powerful and will test a large amount of HPCS,
and the methods are typically able to overfit on any given validation set,
one needs to consider:

- regularization of the hyperparameters (-> hyper-hyper parameters)
- early stopping criteria of the hyperopt procedure using a second stage validation set

.. code-block:: python

    HPC_SEEDS = sample_random_hpcs()
    HPC_optimizer = ...
    splits = make_splits(dataset)
    JobManager = JobManager(slurm_config)

    outer_splits = cross_validation(dataset)
    inner_split = cross_validation(outer_splits)

    for HPC in initial_hpc_seeds():
        for outer_split in outer_splits:
            for inner_split in inner_splits:
                train(model, HPC, inner_split.train)
                evaluate(model, HPC, inner_split.stage1)
                update(HPC, inner_split.stage1)
                evaluate(HPC, inner_split.stage2)
