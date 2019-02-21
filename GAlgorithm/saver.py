
import os
import fnmatch
import json
from pathlib import Path
from functools import singledispatch

#from tests.tsp_bases import City, Route

@singledispatch
def to_serializable(o):
    '''used by default. You can register other functions
    for different input types with attribute:
    @to_serializable.register(CustomClass)'''
    try:
        return o.serialize()
    except:
        return o


def load(fp):
    '''reads in a genetic algorithm saved file and plots the data'''
    with open(fp, 'r') as f:
        data = f.read()

    dct = json.loads(data)
    if '__GeneticAlgorithm__' in dct:
        o = decode_ga(dct)
    elif '__Route__' in dct:
        o = decode_route(dct)
    else:
        o = dct
    return o


def dump(o, fp):
    '''Saves the object *o* to the *fp* filepath'''
    with open(fp, 'w') as f:
        json.dump(o, f, default=to_serializable, indent=4)


def decode_route(dct):
    '''reads in the starter cities from a saved file'''
    cities = []
    for city in dct['_genes']:
        cities.append(City(city['x'], city['y']))
    return cities


def load_from_files(directory, name_pattern) -> list:
    '''returns a tuple of the data loaded from
    all the files found in the directory
    that matched the name pattern'''
    paths = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for name in filenames:
            if fnmatch.fnmatch(name, name_pattern):
                paths.append(Path(dirpath, name))

    runs = []
    for fp in paths:
        runs.append(load(fp))

    return runs


class GASaveObject:
    def __init__(self):
        '''An object to hold the save data'''
        self.best_routes:Route = []
        self.generations:int = None
        self.mutation_rate:int = None
        self.tourny_size:int = None
        self.classifier:int = None
