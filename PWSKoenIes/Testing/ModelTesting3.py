from ModelMain import s_squared
from QuickDataVisualizer import export_array_array_to_csv
from Utils import get_data_from_csv
from SimpleModels import ExponentialPair
from DifferentialModels import LV, CLV, Harissons

# create an instance of the EP model
dt = 0.005  # parameters for the Euler's method algorithm
max_t = 6.6
models = [ExponentialPair(),
          LV(dt, max_t),
          CLV(dt, max_t),
          Harissons(dt, max_t)]

# some parameters
e1 = 0.01  # value of e1 used in the calculating S^2
e2 = 0.025  # value of e2 used in the calculating S^2

datas = [get_data_from_csv("Data/exp4_beker{}_data.csv".format(i), t_scale=1) for i in range(1, 5)]


parameters = [{'a': 0.4351870328161882, 'b': -0.6026012957441343},
              {'alpha': -0.2650388840634637, 'beta': -0.3432609437648339, 'gamma': -0.1432990418040894, 'delta': -0.23409880107336478},
              {'K1': 0.6371447220632337, 'K2': 2.3640645157935207, 'alpha12': -21.22085031442453, 'alpha21': 0.6499022734221718, 'r1': 0.01110877404625115, 'r2': 0.5015939258296289},
              {'z(0)': 0.8073625372153405, 'rho': 1.2059364638798984, 'K': 7.486034039359279, 'omega': 22.709268838129123, 'phi': -0.749210916370186, 'nu': -0.005418103339557228, 'sigma_hat': -13.343040936676141, 'delta': -0.07237514236372071, 'gamma': 0.8701801651909592}]

stuff = []
for model, p in zip(models, parameters):
    stuff.append([round(s_squared(data, model, p, e1, e2), 4) for data in datas])

export_array_array_to_csv(stuff, "../test_result2.csv")




















