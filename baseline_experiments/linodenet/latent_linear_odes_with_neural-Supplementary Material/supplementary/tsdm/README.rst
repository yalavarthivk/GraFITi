Time Series Datasets and Models
================================

This repository contains tools to import important time series datasets and baseline models.

.. note::

    Documentation is hosted at https://bvt-htbd.gitlab-pages.tu-berlin.de/kiwi/tf1/tsdm


Installation guide
------------------

.. code-block:: bash

    poetry shell
    poetry install


Multiple Origins Push
---------------------

The project is located at
 - https://software.ismll.uni-hildesheim.de/ISMLL-internal/special-interest-group-time-series/tsdm
 - https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/tsdm

To push to both repositories do the following

1. Remove all remotes.

.. code-block:: shell

    git remote -v
    git remote remove ...

2. Add origin either berlin or hildesheim

.. code-block:: shell

    git remote add origin https://software.ismll.uni-hildesheim.de/ISMLL-internal/special-interest-group-time-series/tsdm.git
    git remote add origin https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/tsdm.git

3. Tell GIT from which remote to perform pulls for the branch

.. code-block:: shell

    git remote set-url --add --push origin https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/tsdm.git
    git remote set-url --add --push origin https://software.ismll.uni-hildesheim.de/ISMLL-internal/special-interest-group-time-series/tsdm.git
    git branch --set-upstream-to=origin/master master

4. Check if everything is correct. `git remote -v` should print

.. code-block:: shell

    origin  https://software.ismll.uni-hildesheim.de/ISMLL-internal/special-interest-group-time-series/tsdm (fetch)
    origin  https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/tsdm (push)
    origin  https://software.ismll.uni-hildesheim.de/ISMLL-internal/special-interest-group-time-series/tsdm (push)
