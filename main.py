import linecache

from city import City
from genetic_algo import geneticAlgorithmPlot

# url for the pubs dataset
# target_url = "http://www.math.uwaterloo.ca/tsp/pubs/files/uk24727_latlong.txt"

POPULATION_SIZE = 200
ELITE_SIZE = 50
MUTATION_RATE = 0.01
GENERATIONS = 4000


def scrape(linecount=0):
    coordinates = []

    with open("berlin.txt") as reader:
        for line in reader:
            linecount += 1
    # idxs = random.sample(range(linecount), size)
    coordinates = [linecache.getline("berlin.txt", i).strip("\n").split() for i in range(1, linecount + 1)]

    return [(float(line[0]), float(line[1])) for line in coordinates]


if __name__ == '__main__':
    cityList = [City(x=item[0], y=item[1]) for item in scrape()]

    geneticAlgorithmPlot(population=cityList, popSize=POPULATION_SIZE, eliteSize=ELITE_SIZE, mutationRate=MUTATION_RATE, generations=GENERATIONS)

