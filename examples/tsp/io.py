''' Input Output of TSP related objects '''


from .population import (
    City,
    Route, 
)


def read_cities(fp):
    '''reads in the starter cities from a saved file'''
    
    cities = []

    with open(fp, 'r') as f:
        f.readline()
        for line in f:
            xy = line.split('\t')
            x = int(xy[0])
            y = int(xy[1])
            cities.append(City(x,y))

    return cities


def make_cities(numcities: int):
    cities = [City().randomize(0,200,0,200) for i in range(numcities)]
    route = Route(cities)
    return route

def write_cities(numcities: int):
    route = make_cities(numcities)
    route.to_csv(f'cities\\{numcities}cities.txt')
