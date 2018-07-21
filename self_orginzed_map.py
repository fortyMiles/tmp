import numpy as np
import pandas as pd
import random
import time
from kmeans_cluster_points import cluster_points
import matplotlib.pyplot as plt


def select_closest(candidates, origin):
    """Return the index of the closest candidate to a given point."""
    return euclidean_distance(candidates, origin).argmin()


def euclidean_distance(a, b):
    """Return the array of distances of two numpy arrays of points."""
    return np.linalg.norm(a - b, axis=1)


def route_distance(cities):
    """Return the cost of traversing a route of cities in a certain order."""
    points = cities[['x', 'y']]
    distances = euclidean_distance(points, np.roll(points, 1, axis=0))
    return np.sum(distances)


def normalize(points):
    """
    Return the normalized version of a given vector of points.

    For a given array of n-dimensions, normalize each dimension by removing the
    initial offset and normalizing the points in a proportional interval: [0,1]
    on y, maintining the original ratio on x.
    """
    ratio = (points.x.max() - points.x.min()) / (points.y.max() - points.y.min()), 1
    ratio = np.array(ratio) / max(ratio)
    norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    return norm.apply(lambda p: ratio * p, axis=1)


def generate_network(size):
    """
    Generate a neuron network of a given size.

    Return a vector of two dimensional points in the interval [0,1].
    """
    return np.random.rand(size, 2)


def get_neighborhood(center, radix, domain):
    """Get the range gaussian of given radix around a center index."""

    # Impose an upper bound on the radix to prevent NaN and blocks
    if radix < 1:
        radix = 1

    # Compute the circular network distance to the center
    deltas = np.absolute(center - np.arange(domain))
    distances = np.minimum(deltas, domain - deltas)

    # Compute Gaussian distribution around the given center
    return np.exp(-(distances * distances) / (2 * (radix * radix)))


def get_route(cities, network):
    """Return the route computed by a network."""
    cities['winner'] = cities[['x', 'y']].apply(
        lambda c: select_closest(network, c),
        axis=1, raw=True)

    return cities.sort_values('winner').index


def som(problem, iterations, learning_rate=0.8):
    """Solve the TSP using a Self-Organizing Map."""

    # Obtain the normalized set of cities (w/ coord in [0,1])
    cities = problem.copy()

    cities[['x', 'y']] = normalize(cities[['x', 'y']])

    # The population size is 8 times the number of cities
    n = cities.shape[0] * 8

    # Generate an adequate network of neurons:
    network = generate_network(n)
    print('Network of {} neurons created. Starting the iterations:'.format(n))

    for i in range(iterations):
        if not i % 100:
            print('\t> Iteration {}/{}'.format(i, iterations), end="\r")
        # Choose a random city
        city = cities.sample(1)[['x', 'y']].values
        winner_idx = select_closest(network, city)
        # Generate a filter that applies changes to the winner's gaussian
        gaussian = get_neighborhood(winner_idx, n // 10, network.shape[0])
        # Update the network's weights (closer to the city)
        network += gaussian[:, np.newaxis] * learning_rate * (city - network)
        # Decay the variables
        learning_rate = learning_rate * 0.99997
        n = n * 0.9997

        if n < 1:
            print('Radius has completely decayed, finishing execution',
                  'at {} iterations'.format(i))
            break
        if learning_rate < 0.001:
            print('Learning rate has completely decayed, finishing execution',
                  'at {} iterations'.format(i))
            break
    else:
        print('Completed {} iterations.'.format(iterations))

    # plot_network(cities, network, name='diagrams/final.png')

    route = get_route(cities, network)
    # plot_route(cities, route, 'diagrams/route.png')
    return route


def main(start, points, learning_rate=0.8, epoch=100000):
    points = [start] + list(points)
    n = len(points)

    cities = pd.DataFrame.from_dict({
        'city': [str(i) for i in range(n)],
        'x': [p.x for p in points],
        'y': [p.y for p in points]

    })

    route = som(cities, epoch, learning_rate)

    route = list(route)

    start_index = route.index(0)

    route_from_start = route[start_index:] + route[: start_index] + [0]

    locations = points
    return -1,  [locations[i] for i in route_from_start]


def k_person_tsp_using_k_means(k, start, points, cost=None, learning_rate=0.8, max_epoch=100000, draw_solution=False):
    kernels = cluster_points(start, points, kernel=k)
    solution = {}

    for index in kernels:
        print('solving path: {}'.format(index+1))

        points = kernels[index]

        print('  there are {} points:'.format(len(points)))
        route = main(start, points, learning_rate, max_epoch)
        solution[index] = route[1]

    if draw_solution: draw_routes(start, solution)

    return solution


def draw_routes(start, routes):
    for index in routes:
        locations = list(routes[index])
        locations = [start] + locations + [start]
        X = np.array([x for x, y in locations])
        Y = np.array([y for x, y in locations])
        plt.plot(X, Y, '-o')
        plt.plot([start[0]], [start[1]], 'ro')


if __name__ == '__main__':
    from collections import namedtuple

    n = 30

    s = time.time()

    Point = namedtuple('Point', ['x', 'y'])

    start = Point(x=50, y=50)

    fake_cities = [Point(x=random.randrange(100), y=random.randrange(100)) for _ in range(n)]

    print(main(start, fake_cities))

    print('used time: {}'.format(time.time() - s))
