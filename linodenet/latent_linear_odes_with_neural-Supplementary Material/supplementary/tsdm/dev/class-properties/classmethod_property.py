#!/usr/bin/env python

# In[1]:


import sys

print(sys.version)


# In[2]:


from abc import ABC, ABCMeta, abstractmethod
from time import sleep


def compute(obj, s):
    print(f"Computing {s} of {obj} ...", end="")
    sleep(3)
    print("DONE!")
    return "Phew, that was a lot of work!"


class MyMetaClass(ABCMeta):
    @property
    def expensive_metaclass_property(cls):
        """This may take a while to compute!"""
        return compute(cls, "metaclass property")


class MyBaseClass(ABC, metaclass=MyMetaClass):
    @property
    @classmethod
    @property
    def expensive_class_property(cls):
        """This may take a while to compute!"""
        return compute(cls, "class property")

    @property
    def expensive_instance_property(self):
        """This may take a while to compute!"""
        return compute(self, "instance property")


class MyClass(MyBaseClass):
    """Some subclass of MyBaseClass"""


help(MyClass)
