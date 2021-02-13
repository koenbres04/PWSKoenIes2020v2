from ModelMain import s_squared, improved_gradient_descent_step
from QuickDataVisualizer import write_comparison_csv
from Utils import get_data_from_csv
from DifferentialModels import CLV
from AnalyisisMain import fetch_parameters

# create an instance of the EP model
dt = 0.005  # parameters for the Euler's method algorithm
max_t = 6.6
model = CLV(dt, max_t)


parameters = [fetch_parameters("CLVAnalysisResults", False, data_set_num=3, step_num=10874+i, method="AGD")
              for i in range(-4, 3)]

# some parameters
e1 = 0.01  # value of e1 used in the calculating S^2
e2 = 0.025  # value of e2 used in the calculating S^2

datas = [get_data_from_csv("Data/exp4_beker{}_data.csv".format(i), t_scale=1) for i in range(1, 5)]
for p in parameters:
    print(s_squared(datas[2], model, p, e1, e2))
for p in parameters:
    print(improved_gradient_descent_step([datas[2]], model, p, 1e-5, 2e-1, minimal_second_gradient=1e-7, e1=0.01, e2=0.025))

write_comparison_csv("../test_result.csv", datas[2], model, parameters, 0.01)
