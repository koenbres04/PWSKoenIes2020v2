from ModelMain import find_minimal_parameters, s_squared, improved_gradient_descent_step
from QuickDataVisualizer import export_array_array_to_csv
from Utils import get_data_from_csv
from DifferentialModels import CLV
from AnalyisisMain import fetch_parameters

# some parameters
e1 = 0.01  # value of e1 used in the calculating S^2
e2 = 0.025  # value of e2 used in the calculating S^2
epsilon = 0.00001
step_size = 2e-1
debug_mod = 1000

datas = [get_data_from_csv("../Data/exp4_beker{}_data.csv".format(i), t_scale=1) for i in range(1, 5)]

# create an instance of the CLV model
dt = 0.005  # parameters for the Euler's method algorithm
max_t = 6.6
model = CLV(dt, max_t)

# stuff
min_step = 10874
m0 = 1e-7
m_factor = 10**(1/10)
f_min = -30
f_max = 50
data_set_num = 3


parameters = fetch_parameters("../CLVAnalysisResults", False, min_step, "AGD", data_set_num)

errors = []

for f in range(f_min, f_max+1):
    m = m0*(m_factor**f)
    step = improved_gradient_descent_step([datas[data_set_num-1]], model, parameters, epsilon, step_size,
                                          minimal_second_gradient=m, e1=e1, e2=e2)
    p = parameters.copy()
    for key in p.keys():
        p[key] += step[key]
    errors.append(s_squared(datas[data_set_num-1], model, p, e1=e1, e2=e2))

lines = [[m0*(m_factor**f)] + [errors[f-f_min]] for f in range(f_min, f_max+1)]

export_array_array_to_csv(lines, "test_result5.csv")
