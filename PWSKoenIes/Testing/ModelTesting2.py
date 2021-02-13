from ModelMain import s_squared, improved_gradient_descent_step
from QuickDataVisualizer import write_comparison_csv
from Utils import get_data_from_csv
from DifferentialModels import Harissons
from AnalyisisMain import fetch_parameters

# create an instance of the EP model
dt = 0.005  # parameters for the Euler's method algorithm
max_t = 6.6
model = Harissons(dt, max_t)


parameters = [{'z(0)': -0.45223153645116926, 'rho': 0.0007920412829318364, 'K': 0.0009493449635750842, 'omega': 0.0012986046658930506, 'phi': -0.0017495418337219965, 'nu': 0.0009082427975440874, 'sigma_hat': -0.004925397525779543, 'delta': -0.00022202784670470672, 'gamma': -0.0005758564167731943},
              {'z(0)': -8.580080124825165e-06, 'rho': 6.279421592048506e-05, 'K': 5.7803413969339286e-05, 'omega': 2.5434927024463145e-05, 'phi': -4.8882842750110695e-05, 'nu': 1.1599868883423163e-05, 'sigma_hat': -8.142725871441911e-05, 'delta': -1.5168833723879081e-05, 'gamma': -0.0002406132245760127}, {  # initial parameters for the gradient descent algorithms to start searching from
    "z(0)": 0.5,
    "rho": 1,
    "K": 1,
    "omega": 1,
    "phi": 1,
    "nu": 1,
    "sigma_hat": 1,
    "delta": 1,
    "gamma": 1,
}]

# some parameters
e1 = 0.01  # value of e1 used in the calculating S^2
e2 = 0.025  # value of e2 used in the calculating S^2

datas = [get_data_from_csv("Data/exp4_beker{}_data.csv".format(i), t_scale=1) for i in range(1, 5)]
# for p in parameters:
#     print(s_squared(datas[0], model, p, e1, e2))
# for p in parameters:
#     print(improved_gradient_descent_step([datas[0]], model, p, 1e-5, 2e-1, minimal_second_gradient=1e-7, e1=0.01, e2=0.025))

write_comparison_csv("../test_result.csv", datas[0], model, parameters, 0.01)
