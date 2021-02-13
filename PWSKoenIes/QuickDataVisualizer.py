""""
A file for quickly visualising data by printing out graphs made out of text
"""
from ModelMain import Model
import math
from Utils import find_t0_data_point, find_max_t_data_point, unite_data, export_array_array_to_csv
from typing import Union, List, Optional


# print a graph of some data
def print_data(data, width=200):
    x_max = 0
    y_max = 0
    for t, x, y in data:
        if x > x_max:
            x_max = x
        if y > y_max:
            y_max = y
    for t, x, y in data:
        print("{} | ".format(round(t, 2)) + "#"*(0 if x <= 0 else int(round(x/x_max*width, 0))) +
              " <- {}".format(round(x, 2)))
        print(" "*len("{} | ".format(round(t, 2))) + "-"*(0 if y <= 0 else int(round(y/y_max*width, 0)))
              + " <- {}".format(round(y, 2)))


# print a graph of the prediction of a model
def print_model(model: Model, parameters: dict, x0: float, y0: float, max_t: float, dt=1):
    data = model.get_data(parameters, x0, y0, 0, max_t, math.floor(max_t/dt)+1)
    print_data(data)


# writes a .csv file that contains a graph comparing data to the prediction of a model
def write_comparison_csv(file_name: str, data, model: Model, parameters: Union[list, dict], model_dt: float,
                         headers: Optional[List[str]] = None):
    if isinstance(parameters, dict):
        parameters = [parameters]
    x0, y0 = find_t0_data_point(data)
    t_max = find_max_t_data_point(data)[0]
    x_data = [[t, xt] for t, xt, yt in data]
    y_data = [[t, yt] for t, xt, yt in data]
    x_datas = []
    y_datas = []
    for parameters0 in parameters:
        model_data = model.get_data(parameters0, x0, y0, 0, t_max, math.ceil(t_max/model_dt+1))
        x_datas.append([[t, xt] for t, xt, yt in model_data])
        y_datas.append([[t, yt] for t, xt, yt in model_data])
    final_data = x_data
    for model_x_data in x_datas:
        final_data = unite_data(final_data, model_x_data)
    final_data = unite_data(final_data, y_data)
    for model_y_data in y_datas:
        final_data = unite_data(final_data, model_y_data)
    new_headers = None if headers is None else [""]+["x " + h for h in headers]+["y " + h for h in headers]
    export_array_array_to_csv(final_data, file_name, headers=new_headers)
