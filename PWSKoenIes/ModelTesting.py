from ModelMain import gradient_descent_minimal_parameters, gradient_descent_minimal_parameters2, find_minimal_parameters
from DifferentialModels import LV
from QuickDataVisualizer import print_data, print_model
from Utils import get_data_from_csv

# define initial parameters
initial_parameters = {
    "alpha": 2,
    "beta": 0.06,
    "gamma": 0.03,
    "delta": 0.2
}

# create an approximation of the Lotka-Volterra model
model = LV(dt=0.005, max_t=10)

# create data
test_parameters = {
    "alpha": 1,
    "beta": 0.05,
    "gamma": 0.02,
    "delta": 0.14
}
x0 = 20
y0 = 30
# data = model.get_data(test_parameters, x0, y0, 0, 10, 31)

data = get_data_from_csv("exp4_data_1.csv", t_scale=1)

# new_parameters = gradient_descent_minimal_parameters(data, model, initial_parameters, 0.00001, 1e-7, 10000)
new_parameters = gradient_descent_minimal_parameters2(data, model, initial_parameters, 0.00001, 1e1, 10000,
                                                      y_factor=100, debug_mod=100)
# new_parameters = find_minimal_parameters([data], model, initial_parameters, 0.00001, 1e1, 10000,
#                                          y_factor=100, debug_mod=100)

x0 = data[0][1]
y0 = data[0][2]
# print_model(model, test_parameters, x0, y0, 10, 1)
print_data(data)
print(" ")
print_model(model, new_parameters, x0, y0, 7, 1)
