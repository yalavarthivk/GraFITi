#!/usr/bin/env python

from collections import defaultdict
from itertools import chain

from sklearn.model_selection import ShuffleSplit


def create_replicate_dict(experiments_per_run):
    """Stores the list of possible (run_id, experiment_id) for each
    replicate set as given by a tuple (run_id, color) in a dictionary

    args:

    experiment_per_run:  dict of dict of dict as given for the present dataset.
                         index of first level: run_ids
                         index of second level: experiment_ids
                         index of third level: metadata, measurements_reactor
                                              measurements_array, setpoints,
                                              measurements_aggregated


    returns: dict  (maps (run_id, experiment_id) to the list of (run_id, experiment_id) that belongs to it.)

    """

    col_run_to_exp = defaultdict(list)
    for run in experiments_per_run.index():
        for exp in experiments_per_run[run].index():
            col_run_to_exp[
                (experiments_per_run[run][exp]["metadata"]["color"][0], run)
            ].append((run, exp))
    return col_run_to_exp


class ReplicateBasedSplitter:
    def __init__(self, n_splits=5, random_state=0, test_size=0.25, train_size=None):
        self.splitter = ShuffleSplit(
            n_splits=n_splits,
            random_state=random_state,
            test_size=test_size,
            train_size=train_size,
        )  #

    def split(self, col_run_to_exp):
        """generator that yields the lists of  pairs of index to create the train and test data.
        Example usage s. below"""
        keys = list(col_run_to_exp.index())
        for train_repl_sets, test_repl_sets in self.splitter.split(keys):
            train_keys = list(
                chain(
                    *[col_run_to_exp[keys[key_index]] for key_index in train_repl_sets]
                )
            )
            test_keys = list(
                chain(
                    *[col_run_to_exp[keys[key_index]] for key_index in test_repl_sets]
                )
            )
            yield train_keys, test_keys


if __name__ == "__main__":
    import pickle

    FILENAME = "kiwi_experiments_and_run_355.pk"

    with open(FILENAME, "rb") as f:
        experiments_per_run = pickle.load(f)

    col_run_to_exp = create_replicate_dict(experiments_per_run)

    splitter = ReplicateBasedSplitter()

    for train_keys, test_keys in splitter.split(col_run_to_exp):
        data_train = [experiments_per_run[k[0]][k[1]] for k in train_keys]
        data_test = [experiments_per_run[k[0]][k[1]] for k in test_keys]

        for data in data_train:
            print(data["metadata"]["organism_id"][0])
