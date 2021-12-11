import numpy as np
import math
from simulated_annealing import CostFunction, NeighbourhoodFunction, TemperatureFunction
from simulated_annealing.utils import plot_function

class SimulatedAnnealing:

    def __init__(self, costFunction, neighbourhoodFunction, temperatureFunction, inititalTemp = 1, finalTemp = 0.0001) -> None:
        """This class conducts the actual simulated annealing part. All the component functions are
        taken in as parameters and then utlised to find the proposed approximate optimal solution.

        Parameters
        ----------
        costFunction : function
            cost function that we want to minimise
        neighbourhoodFunction : function
            function that finds the neighbour of the current vector in the search space
        temperatureFunction : function
            function that controls how the temperature falls
        inititalTemp : int, optional
            initial temperature value for the annealing, by default 1
        finalTemp : float, optional
            final temperature value for the annealing, by default 0.0001
        """

        self.costFunction = costFunction
        self.neighbourhoodFunction = neighbourhoodFunction
        self.temperatureFunction = temperatureFunction
        self.initialTemp = inititalTemp
        self.finalTemp = finalTemp
        self.currentTemp = self.initialTemp
        self.vector = np.random.random(size=2)

    def begin_process(self) -> tuple:
        """This function conducts the annealing itself where as the temperature decreases,
        we converge slowly towards the optimal solution

        Returns
        -------
        tuple
            Returns the approximate optimal solution found by the algorithm along with the
            value of the objective function
        """

        while self.currentTemp > self.finalTemp:

            # Find a random neighbour and evaluate the cost function for both vectors
            valueOfVector = self.costFunction(self.vector)
            neighbourVector = self.neighbourhoodFunction(self.vector)
            valueOfNeighbour = self.costFunction(neighbourVector)

            # Find the change in the cost function to see which one is a better solution
            deltaE =  valueOfVector - valueOfNeighbour
            # Decrement temperature
            self.currentTemp = self.temperatureFunction(self.currentTemp)

            # Accept the new proposed solution if it is better than the current one; or accept a worse one with some
            # positive probability
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

    costFunction = CostFunction.square_sum
    temperatureFunction = TemperatureFunction.geometric
    neighbourhoodFunction = NeighbourhoodFunction.shift_all_randomly_normal

    SA = SimulatedAnnealing(costFunction, neighbourhoodFunction, temperatureFunction)
    result = SA.begin_process()
    plot_function(costFunction, optima=tuple(result[0]))







