.. toctree::

Nested-Cross-Validation for Hyperparameter Tuning
-------------------------------------------------

Vanilla Grid-Search / Random Search

Without parallelization

.. code-block:: python

    HPCS = create_hyperparameter_configs()

    for outer_split in cross_validation(dataset):
        for HPC in HPCS:
            for inner_split in cross_validation(outer_split):
                model = Model(HPC)
                train(model, split.train)
                iscore[HPC, split] = evaluate(model, split.test)
        HPC_opt = argmax(aggregate(iscore, splits))
        model = Model(HPC_opt)
        train(model, outer_split.train)
        score[model, outer_split] = evaluate(model, outer_split.test)
    return aggregate(score, splits)

With parallelization

.. code-block:: python

    HPCS = create_hyperparameter_configs()

    for outer_split in cross_validation(dataset):
        JobManager.submit(outer_job, HPCS, outer_split)
    results = await JobManager.get_results()
    return aggregate(results)

where

.. code-block:: python

    async def outer_job(HPCS, outer_split) -> HPC:
        for HPC in HPCS:
            for inner_split in cross_validation(outer_split):
                JobManager.submit(HPC, inner_split)
        iscores = await JobManager.get_results()
        JobManager.submit(best_hpc, model, outer_split)
        return await JobManger.get_result()
