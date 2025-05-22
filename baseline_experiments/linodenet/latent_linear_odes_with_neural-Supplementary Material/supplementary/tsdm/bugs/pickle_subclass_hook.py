#!/usr/bin/env python


import pickle
from abc import ABC, abstractmethod


def create_hook(func, hook):
    def wrapper(*args, **kwargs):
        hook()
        return func(*args, **kwargs)

    return wrapper


class Parent(ABC):
    def __init__(self, *args, **kwargs):
        self.parent_name = "SuperClass"
        self.run = create_hook(self.run, self.global_pre_run_hook)

    # global hook to run before each subclass run()
    def global_pre_run_hook(self):
        print("Hooked")

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError()


class Child(Parent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "ChildClass"

    def run(self):
        print(f"my parent name is {self.parent_name}")
        print(f"my name is {self.name}")
        return 22


if __name__ == "__main__":

    obj = Child()
    result = obj.run()
    print(pickle.dumps(obj))
