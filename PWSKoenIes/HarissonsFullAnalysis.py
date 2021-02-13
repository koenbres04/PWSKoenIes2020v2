from DifferentialModels import Harissons
from AnalyisisMain import analise

# file names
data_file_names = ["Data/exp4_beker1_data.csv",
                   "Data/exp4_beker2_data.csv",
                   "Data/exp4_beker3_data.csv",
                   "Data/exp4_beker4_data.csv"]

name = "Harissons"

# constants
dt = 0.005  # parameters for the Euler's method algorithm
max_t = 6.6
analysis_parameters = {
    "step_size11": 5e-2,  # values of eta used for the normal gradient descent algorithm applied to a single dataset
    "step_size12": 2e-1,  # values of eta used for the normal gradient descent algorithm applied to all datasets
    "step_size21": 1e-1,  # values of eta used for the modified gradient descent algorithm applied to a single dataset
    "step_size22": 1e-1,  # values of eta used for the modified gradient descent algorithm applied to all datasets
    "epsilon1": 0.00001,  # epsilon used for the normal gradient descent algorithm
    "epsilon2": 0.00001,  # epsilon used for the modified gradient descent algorithm
    "minimal_second_gradient": 1e-7,   # value of m used for the modified gradient descent algorithm
    "e1": 0.01,  # value of e1 used in the calculating S^2
    "e2": 1/40,  # value of e2 used in the calculating S^2
    "single_data_set_num_steps": 30000,
    "all_data_sets_num_steps": 30000,
}

initial_parameters = {  # initial parameters for the gradient descent algorithms to start searching from
    "z(0)": 0.5,
    "rho": 1,
    "K": 10,
    "omega": 1,
    "phi": 1,
    "nu": -0.01,
    "sigma_hat": 1,
    "delta": -0.5,
    "gamma": 1,
}

# constants for the outputs
debug_mod = 1000  # number that indicates how much is printed in the console for debugging
model_graph_dt = 0.01

# create the model object
model = Harissons(dt=dt, max_t=max_t)

# do the analysis
analise(name, model, initial_parameters, data_file_names, analysis_parameters,
        debug_mod=debug_mod, model_graph_dt=model_graph_dt)
