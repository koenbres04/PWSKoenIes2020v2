from ModelMain import Model
from math import exp


class ExponentialPair(Model):
    def get_prediction(self, parameters: dict, x0: float, y0: float):
        a = parameters["a"]
        b = parameters["b"]

        def x(t):
            return x0*exp(a*t)

        def y(t):
            return y0*exp(b*t)

        return x, y
