CONTRIBUTING
============

Getting started
---------------

1. Fork the GitLab project from https://software.ismll.uni-hildesheim.de/ISMLL-internal/special-interest-group-time-series/tdm.

   Use your personal namespace, e.g. https://software.ismll.uni-hildesheim.de/rscholz/tsdm-dev

2. Clone the forked project locally to your machine. ::

    git clone https://software.ismll.uni-hildesheim.de/rscholz/tsdm-dev
    cd tsdm-dev

3. Setup the virtual environment

   3.1 Via poetry (recommended).::

        pip install --upgrade poetry
        poetry shell
        poetry install

   3.2 Via conda (You may have to rename ``tables`` ⟶ ``pytables`` and ``torch`` ⟶ ``pytorch``).::

        conda create --name tsdm-dev --file requirements.txt
        conda activate tsdm-dev
        conda install --file requirements-dev.txt

   3.3 Via pip.::

        sudo apt install python3.10
        python3.10 -m virtualenv .venv
        . venv/bin/activate
        pip install -e .

   Verify that the installation was successful.::

    python -c "import tsdm"

4. Setup remote repositories and pre-commit hooks.::

    ./run/setup_remote.sh
    ./run/setup_precommit.sh

4. Create a new working branch. Choose a descriptive name for what you are trying to achieve.::

    git checkout -b feature-xyz

5. Write your code, bonus points for also adding unit tests.

   5.1 Write your code in the ``src`` directory.

   5.2 Write your unit tests in the ``tests`` directory.

   5.3 Check if tests are working via ``pytest``.

   5.4 Check for type errors via ``mypy``.

   5.5 Check for style errors via ``flake8``.

   5.6 Check for code quality via ``pylint``.

6. Write descriptive commit messages. Try to keep individual commits easy to understand
   (changing dozens of files, writing 100's of lines of code is not!). ::

    git commit -m '#42: Add useful new feature that does this.'

9. Make sure your changes are parsimonious with the linting and do not break any tests.::

    pip install -r requirements-flake8.txt
    pip install -r requirements-extra.txt

10. Push changes in the branch to your forked repository on GitHub. ::

     git push origin feature-xyz

11. Create a merge request
