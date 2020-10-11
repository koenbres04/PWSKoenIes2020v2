""""
- Koen Bresters 2020
Python file containing the definition of a Model
"""
import abc
from Utils import lerp, find_t0_data_point

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
def s_squared(data, model: Model, parameters, x_factor=1, y_factor=1):
    # find datapoint for t=0
    x0, y0 = find_t0_data_point(data)

    x, y = model.get_prediction(parameters, x0, y0)

    return sum((x(t) - xt)**2*x_factor+(y(t) - yt)**2*y_factor for t, xt, yt in data)/len(data)


# computes a gradient descent step
# parameters: parameter-values to take the step from
# epsilon: a small nonzero number used in approximating the gradient
# step_size: eta in the recursive formula
# num_iterations: number of steps taken
# x_factor: the alpha^2 value used in computing S^2
# y_factor: the beta^2 value used in computing S^2
# error_out: a list that the value of S^2 is appended onto
def gradient_descent_step(data, model: Model, parameters: dict, epsilon: float,
                          step_size: float, x_factor=1, y_factor=1, error_out=None) -> dict:
    # compute s_squared at parameters
    try:
        error = s_squared(data, model, parameters, x_factor, y_factor)
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
            new_error = s_squared(data, model, new_parameters, x_factor, y_factor)
        except ModelException:
            print("Warning: search terminated with an error")
            return parameters

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
# x_factor: the alpha^2 value used in computing S^2
# y_factor: the beta^2 value used in computing S^2
# error_out: a list that the value of S^2 is appended onto
# minimal_second_gradient: the value of m in the recursive formula
def improved_gradient_descent_step(data, model: Model, parameters: dict, epsilon: float,
                                   step_size: float, x_factor=1, y_factor=1,
                                   error_out=None, minimal_second_gradient=0) -> dict:
    # compute s_squared at parameters
    try:
        error = s_squared(data, model, parameters, x_factor, y_factor)
    except ModelException:
        print("Warning: search terminated with an error")
        return parameters

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
            new_error = s_squared(data, model, new_parameters, x_factor, y_factor)
        except ModelException:
            print("Warning: search terminated with an error")
            return parameters
        # compute the S^2 at the slightly changed parameters
        try:
            new_new_error = s_squared(data, model, new_parameters, x_factor, y_factor)
        except ModelException:
            print("Warning: search terminated with an error")
            return parameters

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
# x_factor: the alpha^2 value used in computing S^2
# y_factor: the beta^2 value used in computing S^2
# error_out: a list that the value of S^2 is appended onto after every step
# minimal_second_gradient: the value of m in the recursive formula
def find_minimal_parameters(datas, model: Model, start_parameters: dict, epsilon: float,
                            step_size: float, num_iterations: int, use_improved_descent=True, x_factor=1, y_factor=1,
                            error_out=None, debug_mod=1, minimal_second_gradient=0) -> dict:
    parameters = start_parameters.copy()
    # take num_iteration steps
    for i in range(num_iterations):
        # create a list to store errors in if necessary
        if error_out is not None or DEBUG_MODE:
            errors = []
        else:
            errors = None

        step_sum = {key: 0 for key in parameters.keys()}
        # loop through all datasets
        for data in datas:
            # try to compute the step
            try:
                if use_improved_descent:
                    step = improved_gradient_descent_step(data, model, parameters, epsilon, step_size, x_factor,
                                                          y_factor, errors, minimal_second_gradient)
                else:
                    step = gradient_descent_step(data, model, parameters, epsilon, step_size, x_factor, y_factor,
                                                 errors)
            except StepException:
                print("Warning: search terminated with an error")
                return parameters

            # add the step to the sum
            for key, value in step.items():
                step_sum[key] += value
        # take the averaged step
        for key in parameters.keys():
            parameters[key] += step_sum[key]/len(datas)

        # print debugging values in case debug mode is enabled
        if DEBUG_MODE:
            if i % debug_mod == 0:
                print("step: {}  |  error: {}  |  step: {}".format(i, sum(errors),
                                                                   {key: step_sum[key]/len(datas)
                                                                    for key in step_sum.keys()}))
                print(parameters)

        # store the sum of the errors if necessary
        if error_out is not None:
            error_out.append(sum(errors))

    # add final error-value
    if error_out is not None:
        errors = [s_squared(data, model, parameters, x_factor=x_factor, y_factor=y_factor) for data in datas]
        error_out.append(sum(errors))

    # return minimised parameters
    return parameters


# finds a set of parameters that locally minimize S^2 in the neighbourhood of start_parameters using gradient descent
# start_parameters: starting point of the search
# epsilon: a small nonzero number used in approximating the gradient
# step_size: eta in the recursive formula
# num_iterations: number of steps taken
# x_factor: the alpha^2 value used in computing S^2
# y_factor: the beta^2 value used in computing S^2
# error_out: a list that the value of S^2 is appended onto after every step
def gradient_descent_minimal_parameters(data, model: Model, start_parameters: dict, epsilon: float,
                                        step_size: float, num_iterations: int, x_factor=1, y_factor=1,
                                        error_out=None, debug_mod=1) -> dict:
    parameters = start_parameters.copy()
    # take num_iteration steps
    for i in range(num_iterations):
        # compute s_squared at parameters
        try:
            error = s_squared(data, model, parameters, x_factor, y_factor)
        except ModelException:
            print("Warning: search terminated with an error")
            return parameters

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
                new_error = s_squared(data, model, new_parameters, x_factor, y_factor)
            except ModelException:
                print("Warning: search terminated with an error")
                return parameters

            # store the approximated rate of change in the gradient dictionary
            gradient[key] = (new_error-error)/epsilon

        # take a step proportional to negative the gradient and step_size
        for key, value in parameters.items():
            parameters[key] += -gradient[key] * step_size

        # print debugging values in case debug mode is enabled
        if DEBUG_MODE:
            if i % debug_mod == 0:
                print("step: {}  |  error: {}  |  gradient: {}".format(i, error, gradient))
                print(parameters)

    # add final error to error_out
    if error_out is not None:
        # compute s_squared at parameters
        try:
            error = s_squared(data, model, parameters, x_factor, y_factor)
        except ModelException:
            print("Warning: search terminated with an error")
            return parameters
        error_out.append(error)

    # return parameters that locally minimize S^2
    return parameters


# finds a set of parameters that locally minimize S^2 in the neighbourhood of start_parameters using the improved
# gradient descent algorithm
# start_parameters: starting point of the search
# epsilon: a small nonzero number used in approximating the gradient
# step_size: eta in the recursive formula
# num_iterations: number of steps taken
# x_factor: the alpha^2 value used in computing S^2
# y_factor: the beta^2 value used in computing S^2
# error_out: a list that the error is appended onto after every step
# minimal_second_gradient: the value of m in the recursive formula
def gradient_descent_minimal_parameters2(data, model: Model, start_parameters: dict, epsilon: float,
                                         step_size: float, num_iterations: int, x_factor=1, y_factor=1,
                                         error_out=None, debug_mod=1, minimal_second_gradient=0) -> dict:
    parameters = start_parameters.copy()
    # take num_iteration steps
    for i in range(num_iterations):
        # compute s_squared at parameters
        try:
            error = s_squared(data, model, parameters, x_factor, y_factor)
        except ModelException:
            print("Warning: search terminated with an error")
            return parameters

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
                new_error = s_squared(data, model, new_parameters, x_factor, y_factor)
            except ModelException:
                print("Warning: search terminated with an error")
                return parameters
            # compute the S^2 at the slightly changed parameters
            try:
                new_new_error = s_squared(data, model, new_parameters, x_factor, y_factor)
            except ModelException:
                print("Warning: search terminated with an error")
                return parameters

            # store the approximated rate of change in the gradient dictionary
            gradient[key] = (new_error-error)/epsilon
            # store the approximated second order derivative in the second gradient dictionary
            second_gradient[key] = ((new_new_error-new_error) - (new_error-error))/epsilon/epsilon

        # take a step proportional to negative the gradient and step_size
        for key, value in parameters.items():
            parameters[key] += -gradient[key] * step_size / max(abs(second_gradient[key]), minimal_second_gradient)

        # print debugging values in case debug mode is enabled
        if DEBUG_MODE:
            if i % debug_mod == 0:
                print("step: {}  |  error: {}  |  gradient: {}".format(i, error, gradient))
                print(parameters)

    # add final error to error_out
    if error_out is not None:
        # compute s_squared at parameters
        try:
            error = s_squared(data, model, parameters, x_factor, y_factor)
        except ModelException:
            print("Warning: search terminated with an error")
            return parameters
        error_out.append(error)

    # return parameters that locally minimize S^2
    return parameters
