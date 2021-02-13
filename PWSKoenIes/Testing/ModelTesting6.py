from ModelMain import s_squared, improved_gradient_descent_step
from QuickDataVisualizer import write_comparison_csv
from Utils import get_data_from_csv, find_t0_data_point
from DifferentialModels import Harissons
from AnalyisisMain import fetch_parameters

# create an instance of the EP model
dt = 0.005  # parameters for the Euler's method algorithm
max_t = 6.6
model = Harissons(dt, max_t)


parameters = [fetch_parameters("../HarissonsAnalysisResults", False, data_set_num=2, step_num=1853, method="AGD")]
# some parameters
e1 = 0.01  # value of e1 used in the calculating S^2
e2 = 0.025  # value of e2 used in the calculating S^2

datas = [get_data_from_csv("../Data/exp4_beker{}_data.csv".format(i), t_scale=1) for i in range(1, 5)]
step = improved_gradient_descent_step([datas[1]], model, parameters[0], 1e-5, 1e-1, minimal_second_gradient=1e-7, e1=0.01, e2=0.025)

parameters.append(parameters[0].copy())
for key in parameters[0].keys():
    parameters[-1][key] = parameters[0][key] + step[key]
for p in parameters:
    print(p)


write_comparison_csv("test_result7.csv", datas[1], model, parameters, 0.01)
