""""
- Koen Bresters 2020
Python file containing the definition of a DifferentialModel and some examples of these
"""
from ModelMain import Model, ModelException
import abc
import math
from Utils import lerp


# the superclass for models that are built from a set of differential equation
class DifferentialModel(Model):
    # dt: gives the step size used in solving the differential equations using Euler's method
    # max_t: time value up to which the amounts of predator and prey are computed
    def __init__(self, dt: float, max_t: float):
        self.dt = dt
        self.max_t = max_t
        self.bonus_variable_names = self.get_bonus_variable_names()

    # a method that returns a list containing the names of variables other than x and y
    @staticmethod
    @abc.abstractmethod
    def get_bonus_variable_names() -> list:
        pass

    # a function that returns the derivative of all variables
    # t: time
    # x: number of prey
    # y: number of predators
    # bonus_variables: dictionary containing the values of any other variables
    # returns: a tuple containing the dervative of x, the derivative of y and a dictionary containing the derivatives
    # of the other variables
    @abc.abstractmethod
    def dv_dt(self, parameters: dict, t: float, x: float, y: float, bonus_variables: dict) -> tuple:
        pass

    def get_prediction(self, parameters: dict, x0: float, y0: float):
        x_values = [x0]
        y_values = [y0]
        bonus_values = {name: parameters[name + "(0)"] for name in self.bonus_variable_names}
        time = 0

        for i in range(math.ceil(self.max_t/self.dt)):
            # compute derivative at t
            try:
                dx_dt, dy_dt, d_bonus_dt = self.dv_dt(parameters, time, x_values[-1], y_values[-1], bonus_values)
            except (FloatingPointError, ArithmeticError):
                raise ModelException("Failed to compute the derivatives at t={}".format(time))

            # add changed values to the list of values
            x_values.append(x_values[-1] + dx_dt*self.dt)
            y_values.append(y_values[-1] + dy_dt*self.dt)
            for name in self.bonus_variable_names:
                bonus_values[name] += d_bonus_dt[name]*self.dt
            # increment time
            time += self.dt

        # the function that gives the model's prediction of the number of predators at each point in time
        def x(t):
            # check whether the value of t is not too large or too small
            if t < 0 or t > self.max_t:
                raise ModelException("t={} out of bounds".format(t))
            # find the indices of the nearest points in time for which the amount was computed
            index = math.floor(t/self.dt)
            next_index = math.ceil(t/self.dt)
            # interpolate
            if index == next_index:
                return x_values[index]
            else:
                return lerp(x_values[index], x_values[index+1], t/self.dt-index)

        # the function that gives the model's prediction of the number of prey at each point in time
        def y(t):
            # check whether the value of t is not too large or too small
            if t < 0 or t > self.max_t:
                raise ModelException("t={} out of bounds".format(t))
            # find the indices of the nearest points in time for which the amount was computed
            index = math.floor(t / self.dt)
            next_index = math.ceil(t / self.dt)
            # interpolate
            if index == next_index:
                return y_values[index]
            else:
                return lerp(y_values[index], y_values[index+1], t/self.dt-index)

        return x, y


# an implementation of the Lotka-Volterra model
class LV(DifferentialModel):
    @staticmethod
    def get_bonus_variable_names() -> list:
        return []

    def dv_dt(self, parameters: dict, t: float, x: float, y: float, bonus_variables: dict) -> tuple:
        alpha = parameters["alpha"]
        beta = parameters["beta"]
        gamma = parameters["gamma"]
        delta = parameters["delta"]

        dx_dt = x*(alpha - beta*y)
        dy_dt = y*(gamma*x - delta)

        return dx_dt, dy_dt, dict()


# an implementation of the Competitive Lotka-Volterra model
class CLV(DifferentialModel):
    @staticmethod
    def get_bonus_variable_names() -> list:
        return []

    def dv_dt(self, parameters: dict, t: float, x: float, y: float, bonus_variables: dict) -> tuple:
        r1 = parameters["r1"]
        r2 = parameters["r2"]
        alpha12 = parameters["alpha12"]
        alpha21 = parameters["alpha21"]
        K1 = parameters["K1"]
        K2 = parameters["K2"]

        dx_dt = r1*x*(1-(x+alpha12*y)/K1)
        dy_dt = r2*y*(1-(y+alpha21*x)/K2)

        return dx_dt, dy_dt, dict()


# an implementation of Harissons model
class Harissons(DifferentialModel):
    @staticmethod
    def get_bonus_variable_names() -> list:
        return ["z"]

    def dv_dt(self, parameters: dict, t: float, x: float, y: float, bonus_variables: dict) -> tuple:
        z = bonus_variables["z"]
        rho = parameters["rho"]
        K = parameters["K"]
        omega = parameters["omega"]
        phi = parameters["phi"]
        nu = parameters["nu"]
        sigma_hat = parameters["sigma_hat"]
        delta = parameters["delta"]  # this is actually -(the delta from the mathematical model)
        gamma = parameters["gamma"]

        f_x = omega*x/(phi+x)*(1-(1+nu*x)*math.exp(-nu*x))

        dx_dt = rho * (1-x/K)*x-f_x*y
        dz_dt = sigma_hat*f_x*y + delta*z
        dy_dt = z - gamma*y

        return dx_dt, dy_dt, {"z": dz_dt}
