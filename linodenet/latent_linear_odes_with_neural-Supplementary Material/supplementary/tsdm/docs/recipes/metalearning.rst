MetaLearning
------------

.. code-block:: python

    for batch_of_tasks in training_tasks:
        for task in batch_of_task:
            train(model, task)

        for task in test_tasks:
            test(model, task)
