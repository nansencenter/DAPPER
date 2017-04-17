#!/usr/bin/env python

#__________________________________________________
# /tools/python/
# pickling.py
#__________________________________________________
# author        : alban.farchi@enpc.fr
# last modified : 2016/11/14
#__________________________________________________
#
# pickle/unpickle
#

from pickle import Pickler, Unpickler

#__________________________________________________

def fromfile(t_file_name):
    with open(t_file_name, 'rb') as f:
        p = Unpickler(f)
        r = p.load()
    return r

#__________________________________________________

def tofile(t_file_name, t_object):
    with open(t_file_name, 'wb') as f:
        p = Pickler(f, protocol = -1)
        p.dump(t_object)

#__________________________________________________

