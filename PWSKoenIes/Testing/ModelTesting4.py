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
min_step = 25650
max_step = 25750
# min_step = 10874
# max_step = 10900
m_values = [1e-8, 0.8e-7, 0.9e-7, 0.98e-7, 0.99e-7, 1e-7, 1.01e-7, 1.02e-7, 1.1e-7, 1.2e-7, 1e-6, 1e-5, 1e-4]
data_set_num = 2


parameters = fetch_parameters("../CLVAnalysisResults", False, min_step, "AGD", data_set_num)

errors = [[] for _ in m_values]

for i, m in enumerate(m_values):
    find_minimal_parameters([datas[data_set_num-1]], model, parameters, epsilon, step_size, max_step-min_step,
                            debug_mod=debug_mod, e1=e1, e2=e2,
                            minimal_second_gradient=m, error_out=errors[i], use_improved_descent=True)

lines = [[i] + [errors[j][i] for j in range(len(errors))] for i in range(max_step-min_step+1)]

export_array_array_to_csv(lines, "test_result6.csv", [""] + ["m={:.2E}".format(m) for m in m_values])
