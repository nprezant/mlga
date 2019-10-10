
import os
import fnmatch
import json
from pathlib import Path
from functools import singledispatch

@singledispatch
def to_serializable(o):
    '''used by default. You can register other functions
    for different input types with attribute:
    @to_serializable.register(CustomClass)'''
    try:
        return o.serialize()
    except:
        return o


def dump(o, fp):
    '''Saves the object *o* to the *fp* filepath'''
    with open(fp, 'w') as f:
        json.dump(o, f, default=to_serializable, indent=4)