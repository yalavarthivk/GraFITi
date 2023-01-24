#!/usr/bin/env python3.9

# In[1]:


def compute(s):
    print(f"Computing {s}...")
    return f"{s} result"


class MethodType:
    "Emulate PyMethod_Type in Objects/classobject.c"

    def __init__(self, func, obj):
        self.__func__ = func
        self.__self__ = obj

    def __call__(self, *args, **kwargs):
        func = self.__func__
        obj = self.__self__
        return func(obj, *args, **kwargs)


def issubclassable(cls):
    try:

        class _(cls):
            ...

    except:
        return False
    return True


class BaseDecorator:
    def __init__(self, obj):
        if hasattr(obj, "__func__"):
            self.__func__ = obj.__func__
        else:  # was never decorated before.
            self.__func__ = obj

    def __call__(self, *args, **kwargs):
        return self.__func__.__call__(*args**kwargs)


def Property(func):
    class Wrapped(property):
        @property
        def __func__(self):
            return self.fget

    return Wrapped(func)


def ClassMethod(func):
    if issubclassable(type(func)):

        class Wrapped(type(func)):
            def __get__(self, obj, cls=None):
                if cls is None:
                    cls = type(obj)
                if hasattr(type(self.__func__), "__get__"):
                    return self.__func__.__get__(cls)
                return MethodType(self.__func__, cls)

        return Wrapped(func)
    return classmethod(func)


# In[2]:


class A:
    # python 3.9

    @classmethod
    @property
    def clsproperty(cls):
        """A python 3.9 class-property"""
        return compute(f"{cls.__name__}'s clsproperty")

    @classmethod
    def clsmethod(cls):
        """A python 3.9 class-method"""
        return compute(f"{cls.__name__}'s clsmethod")

    @property
    def instproperty(self):
        """A python 3.9 instance-property"""
        return compute(f"{self}'s instproperty")

    # our modified versions

    @ClassMethod
    @Property
    def myclsproperty(cls):
        """A custom class-property"""
        return compute(f"{cls.__name__}'s myclsproperty")

    @ClassMethod
    def myclsmethod(cls):
        """A custom classmethod"""
        return compute(f"{cls.__name__}'s myclsmethod")

    @Property
    def myinstproperty(self):
        """A custom instance-property"""
        return compute(f"{self}'s myinstproperty")


# In[3]:

print(A.__dict__)
help(A)
