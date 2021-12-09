import numpy as np
import math
from simulated_annealing import CostFunction, NeighbourhoodFunction, TemperatureFunction

class SimulatedAnnealing:

    def __init__(self, costFunction, neighbourhoodFunction, temperatureFunction, inititalTemp = 1, finalTemp = 0.5) -> None:

        self.costFunction = costFunction
        self.neighbourhoodFunction = neighbourhoodFunction
        self.temperatureFunction = temperatureFunction
        self.initialTemp = inititalTemp
        self.finalTemp = finalTemp
        self.currentTemp = self.initialTemp
        self.vector = np.random.random(size=5)

    def begin_process(self) -> tuple:

        while self.currentTemp > self.finalTemp:

            valueOfVector = self.costFunction(self.vector)
            neighbourVector = self.neighbourhoodFunction(self.vector)
            valueOfNeighbour = self.costFunction(neighbourVector)

            deltaE =  valueOfVector - valueOfNeighbour
            self.currentTemp = self.temperatureFunction(self.currentTemp)

            prob = 1 if deltaE >= 0 else math.exp((deltaE) / self.currentTemp)

            if prob == 1:
                self.vector = neighbourVector

            else:
                rndm = np.random.rand()
                if rndm < prob:
                    self.vector = neighbourVector

                else: continue

        else:
            return (self.vector, self.costFunction(self.vector))

if __name__ == '__main__':

    costFunction = CostFunction.odd_point_max
    temperatureFunction = TemperatureFunction.geometric
    neighbourhoodFunction = NeighbourhoodFunction.shift_all_randomly_normal

    SA = SimulatedAnnealing(costFunction, neighbourhoodFunction, temperatureFunction)
    result = SA.begin_process()
    print(result)





