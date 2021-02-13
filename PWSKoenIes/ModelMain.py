""""
- Koen Bresters 2020
Python file containing the definition of a Model
"""
import abc
from Utils import lerp, find_t0_data_point
import math

# a boolean indicating whether the code is in debug mode
# when enabled, many
DEBUG_MODE = True


class ModelException(Exception):
    """Raised when a model failed to compute a prediction for a specific set of parameters"""
    pass


class StepException(Exception):
    """Raised when a gradient descent step failed"""
    pass


# the abstract superclass for all models
class Model(metaclass=abc.ABCMeta):

    # returns a pair of functions that take in points in time and return predictions for the amounts of
    # predator and prey. Should raise a ModelException when something when the parameters or initial conditions are
    # invalid
    # parameters: a dictionary containing the values of all parameters
    @abc.abstractmethod
    def get_prediction(self, parameters: dict, x0: float, y0: float):
        pass

    # returns a list with 'subdivisions' sample points in time between min_t and max_t
    def get_data(self, parameters: dict, x0: float, y0: float, min_t: float, max_t: float, subdivisions: int) -> list:
        x, y = self.get_prediction(parameters, x0, y0)
        data = []
        for i in range(subdivisions):
            t = lerp(min_t, max_t, i/(subdivisions-1))
            data.append((t, x(t), y(t)))
        return data


# compute S^2 for 'model' and 'data' with 'parameters' as the values for the parameters
# data: list of tuples containing the time, the amount of prey and the amount of predators at that time
def s_squared(data, model: Model, parameters, e1, e2):
    # find datapoint for t=0
    x0, y0 = find_t0_data_point(data)

    x, y = model.get_prediction(parameters, x0, y0)

    try:
        return sum((math.log(max(x(t), 0)+e1)-math.log(max(xt, 0)+e1))**2 +
                   (math.log(max(y(t), 0)+e2)-math.log(max(yt, 0)+e2))**2 for t, xt, yt in data)/len(data)
    except ValueError:
        raise ModelException("a value below e1 or e2 occurred!")


# computes a gradient descent step
# parameters: parameter-values to take the step from
# epsilon: a small nonzero number used in approximating the gradient
# step_size: eta in the recursive formula
# num_iterations: number of steps taken
# e1: value of e_1 used in computing S^2
# e2: value of e_2 used in computing S^2
# error_out: a list that the value of S^2 is appended onto
def gradient_descent_step(datas, model: Model, parameters: dict, epsilon: float,
                          step_size: float, error_out=None, e1=0.01, e2=0.01) -> dict:
    # compute s_squared at parameters
    try:
        error = sum(s_squared(data, model, parameters, e1, e2) for data in datas)
    except ModelException:
        raise StepException()

    # add error to error_out
    if error_out is not None:
        error_out.append(error)

    # compute the gradient of S squared at parameters
    gradient = dict()
    for key, value in parameters.items():
        # create a copy of parameters and change the parameter with key 'key' and change it by epsilon
        new_parameters = parameters.copy()
        new_parameters[key] += epsilon
        # compute the S^2 at the slightly changed parameters
        try:
            new_error = sum(s_squared(data, model, new_parameters, e1, e2) for data in datas)
        except ModelException:
            raise StepException()

        # store the approximated rate of change in the gradient dictionary
        gradient[key] = (new_error - error) / epsilon

    # create and return the step vector
    result = dict()
    for key, value in parameters.items():
        result[key] = -gradient[key] * step_size
    return result


# computes a modified gradient descent step
# parameters: parameter-values to take the step from
# epsilon: a small nonzero number used in approximating the gradient
# step_size: eta in the recursive formula
# num_iterations: number of steps taken
# e1: value of e_1 used in computing S^2
# e2: value of e_2 used in computing S^2
# error_out: a list that the value of S^2 is appended onto
# minimal_second_gradient: the value of m in the recursive formula
def improved_gradient_descent_step(datas, model: Model, parameters: dict, epsilon: float,
                                   step_size: float,
                                   error_out=None, minimal_second_gradient=0, e1=0.01, e2=0.01) -> dict:
    # compute s_squared at parameters
    try:
        error = sum(s_squared(data, model, parameters, e1, e2) for data in datas)
    except ModelException:
        raise StepException()

    # add error to error_out
    if error_out is not None:
        error_out.append(error)

    # compute the gradient of S squared at parameters
    gradient = dict()
    second_gradient = dict()
    for key, value in parameters.items():
        # create a copy of parameters and change the parameter with key 'key' and change it by epsilon
        new_parameters = parameters.copy()
        new_parameters[key] += epsilon
        # compute the S^2 at the slightly changed parameters
        try:
            new_error = sum(s_squared(data, model, new_parameters, e1, e2) for data in datas)
        except ModelException:
            raise StepException()
        new_parameters[key] += epsilon
        # compute the S^2 at the twice slightly changed parameters
        try:
            new_new_error = sum(s_squared(data, model, new_parameters, e1, e2) for data in datas)
        except ModelException:
            raise StepException()

        # store the approximated rate of change in the gradient dictionary
        gradient[key] = (new_error - error) / epsilon
        # store the approximated second order derivative in the second gradient dictionary
        second_gradient[key] = ((new_new_error - new_error) - (new_error - error)) / epsilon / epsilon

    # create and return the step vector
    result = dict()
    for key, value in parameters.items():
        result[key] = -gradient[key] * step_size / max(abs(second_gradient[key]), minimal_second_gradient)
    return result


# finds a set of parameters that locally minimize the sum of S^2 over all datasets in datas in the neighbourhood of
# parameters
# datas: a list of datasets
# start_parameters: starting point of the search
# epsilon: a small nonzero number used in approximating the gradient
# step_size: eta in the recursive formula
# num_iterations: number of steps taken
# e1: value of e_1 used in computing S^2
# e2: value of e_2 used in computing S^2
# error_out: a list that the value of S^2 is appended onto after every step
# minimal_second_gradient: the value of m in the recursive formula
def find_minimal_parameters(datas, model: Model, start_parameters: dict, epsilon: float,
                            step_size: float, num_iterations: int, use_improved_descent=True,
                            error_out=None, parameter_out=None, debug_mod=1,
                            minimal_second_gradient=0, e1=0.01, e2=0.01) -> dict:
    parameters = start_parameters.copy()

    # add first parameters to parameters_out
    if parameter_out is not None:
        parameter_out.append(list(parameters.values()))

    # create an error list
    if error_out is None:
        error_out = []

    # parameters that
    minimal_parameters = parameters.copy()
    min_error = None
    # take num_iteration steps
    for i in range(num_iterations):
        # try to compute the step
        try:
            if use_improved_descent:
                step = improved_gradient_descent_step(datas, model, parameters, epsilon, step_size,
                                                      error_out, minimal_second_gradient, e1, e2)
            else:
                step = gradient_descent_step(datas, model, parameters, epsilon, step_size, error_out, e1, e2)
        except StepException:
            print("Warning: search terminated with an error")
            return minimal_parameters

        error = error_out[-1]

        # check if the error after the last step is the lowest one yet
        if min_error is None or error < min_error:
            minimal_parameters = parameters.copy()
            min_error = error

        # take the averaged step
        for key in parameters.keys():
            parameters[key] += step[key]

        # print debugging values in case debug mode is enabled
        if DEBUG_MODE:
            if i % debug_mod == 0:
                print("step: {}  |  error: {}  |  step: {}".format(i, error,
                                                                   {key: step[key] for key in parameters.keys()}))
                print(parameters)

        # store the parameters if neccessary
        if parameter_out is not None:
            parameter_out.append(list(parameters.values()))

    error = sum([s_squared(data, model, parameters, e1, e2) for data in datas])
    # do a final check if the error currently found is the lowest one yet
    if min_error is None or error < min_error:
        minimal_parameters = parameters.copy()
    # add final error-value
    if error_out is not None:
        error_out.append(error)

    # return minimised parameters
    return minimal_parameters
