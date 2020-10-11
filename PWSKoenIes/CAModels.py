from ModelMain import Model, ModelException
from Utils import lerp
import numpy as np
import math
import random


# An implementation of the CA model described in the PWS
class CAModel(Model):
    # shape: a tuple giving the width, height, depth, etc. of the grid
    # num_runs: amount of runs the average is taken over
    def __init__(self, shape, num_runs, max_t, N, steps_per_time_is_parameter=True,
                 x_factor_is_parameter=True, y_factor_is_parameter=True,
                 steps_per_time=None, x_factor=None, y_factor=None):
        self.shape = shape
        self.num_runs = num_runs
        self.max_t = max_t
        self.N = N
        self.steps_per_time_is_parameter = steps_per_time_is_parameter
        self.x_factor_is_parameter = x_factor_is_parameter
        self.y_factor_is_parameter = y_factor_is_parameter
        self.steps_per_time = steps_per_time
        self.x_factor = x_factor
        self.y_factor = y_factor

    def get_prediction(self, parameters: dict, x0: float, y0: float):
        num_cells = math.prod(self.shape)
        steps_per_time = parameters["steps_per_time"] if self.steps_per_time_is_parameter else self.steps_per_time
        x_factor = parameters["x_factor"] if self.x_factor_is_parameter else self.x_factor
        y_factor = parameters["y_factor"] if self.y_factor_is_parameter else self.y_factor
        num_steps = math.ceil(self.max_t*steps_per_time)

        # compute chances for each cell to be a predator or a prey animal in the initial grid
        x_chance = x0/x_factor
        y_chance = y0/y_factor

        result = np.zeros((num_steps, 2), dtype=float)

        for i in range(self.num_runs):
            num_x = 0
            num_y = 0
            random_grid = np.random.rand(*self.shape)
            grid = np.random.randint(0, 3, self.shape, int)
            # loop through all points
            it = np.nditer(random_grid, flags=["multi_index"])
            while not it.finished:
                value = random_grid[it.multi_index]
                if value < x_chance:
                    grid[it.multi_index] = 1
                    num_x += 1
                elif value < x_chance + y_chance:
                    grid[it.multi_index] = 2
                    num_y += 1
                it.iternext()
            del random_grid

            for j in range(num_steps):
                delta_x, delta_y = self.get_step_changes(grid, parameters)
                num_x += delta_x
                num_y += delta_y
                result[j, 0] = num_x/num_cells*x_factor
                result[j, 1] = num_y/num_cells*y_factor

        result /= self.num_runs

        # define functions to return
        def x(t):
            # check whether the value of t is not too large or too small
            if t < 0 or t > self.max_t:
                raise ModelException("t={} out of bounds".format(t))
            # find the indices of the nearest points in time for which the amount was computed
            index = math.floor(t*steps_per_time)
            next_index = math.ceil(t*steps_per_time)
            # interpolate
            if index == next_index:
                return result[index, 0]
            else:
                return lerp(result[index, 0], result[index+1, 0], t*steps_per_time-index)

        def y(t):
            # check whether the value of t is not too large or too small
            if t < 0 or t > self.max_t:
                raise ModelException("t={} out of bounds".format(t))
            # find the indices of the nearest points in time for which the amount was computed
            index = math.floor(t*steps_per_time)
            next_index = math.ceil(t*steps_per_time)
            # interpolate
            if index == next_index:
                return result[index, 1]
            else:
                return lerp(result[index, 1], result[index+1, 1], t*steps_per_time-index)

        return x, y

    def get_step_changes(self, grid, parameters):
        delta_x = 0
        delta_y = 0

        for _ in range(self.N):
            # generate random cell and a neighbour
            cell_index = tuple(random.randrange(0, m) for m in self.shape)
            neighbour_axis = random.randint(0, len(self.shape))
            neighbour_sign = -1 if random.getrandbits(1) else 1
            neighbour_index = tuple((cell_index[i]+neighbour_sign) % self.shape[i]
                                    if i == neighbour_axis else cell_index[i] for i in range(len(self.shape)))

            # perform the algorithm
            if grid[cell_index] == 2 and grid[neighbour_index] == 1:  # fox eats rabbit
                grid[neighbour_index] = 0
                delta_x -= 1
                if random.random() <= parameters["sigma_f"]:
                    grid[neighbour_index] = 2
                    delta_y += 1
            elif grid[cell_index] == 2 and random.random() <= parameters["p_f"]:  # fox dies
                grid[cell_index] = 0
                delta_y -= 1
            elif grid[cell_index] == 2:  # fox moves to vacancy
                grid[cell_index] = 0
                grid[neighbour_index] = 2
            elif grid[cell_index] == 1 and grid[neighbour_index] == 0:  # rabbit moves or reproduces into vacancy
                if random.random() <= parameters["mu"]:  # rabbit reproduces
                    grid[neighbour_index] = 1
                    delta_x += 1
                else:  # rabbit moves
                    grid[cell_index] = 0
                    grid[neighbour_index] = 1

        return delta_x, delta_y
