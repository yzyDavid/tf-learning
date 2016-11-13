#!/usr/bin/python

'''
this code file is intended to test some python core language specifies.
which may be used then.
'''

import os
import types as tp
import functools as ft
from enum import Enum, unique
import sys
import itertools
import asyncio

lent = 12

@unique
class PyEnum(Enum):
    ASD = 0
    ZXC = 1
    QWE = 2

class PyClass(object):
    def __init__(self):
        self.field_a = 1
        self.__field_private = 'private'
        self.length = None
        self._fc = 5

        global lent
        lent = 13

    def __len__(self):
        return len(self.__field_private)

    lent = 14

    @property
    def fc(self):
        return self._fc

    @fc.setter
    def fc(self, value):
        self._fc = value

    def __eq__(self, other): return True

class PyInherited(PyClass):
    pass

def print_space(lines = 2):
    for _ in range(lines): print()

def foo_args(arg_1, arg_2=0, *args, **kwargs):
    print(args)
    print(kwargs)

foo_partial = ft.partial(foo_args, arg_1=0, arg_2=0)

def func_async():
    for _ in range(10):
        hello()

async def hello(): print('Hello from async')

def main():
    func_async()

    print_space()

    c = PyClass
    print(c)
    d = PyClass()
    print(d)
    e = PyClass()
    setattr(e, 'field_b', 3)
    f = PyInherited()
    print(isinstance(f, PyClass))
    print(e)
    print(type(e))
    print(d.field_a)
    print(e.field_b)
    print(d._PyClass__field_private)
    print(lent)

    print_space()
    
    print(dir(c))
    print(dir(d))
    
    print_space()
    
    foo_args(1, 2, 4, 6, arg_x='12')
    args = (1, 2, 4, 6)
    kw = {'arg_x': '12'}
    foo_args(*args, **kw)
    print(PyEnum.QWE)
    print(PyClass() == PyClass())

if __name__ == '__main__':
    main()

