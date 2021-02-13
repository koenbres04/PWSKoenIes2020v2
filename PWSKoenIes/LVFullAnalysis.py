from DifferentialModels import LV
from AnalyisisMain import analise

# file names
data_file_names = ["Data/exp4_beker1_data.csv",
                   "Data/exp4_beker2_data.csv",
                   "Data/exp4_beker3_data.csv",
                   "Data/exp4_beker4_data.csv"]

name = "LV"

# constants
dt = 0.005  # parameters for the Euler's method algorithm
max_t = 6.6
analysis_parameters = {
    "step_size11": 2e-5,  # values of eta used for the normal gradient descent algorithm applied to a single dataset
    "step_size12": 2e-5,  # values of eta used for the normal gradient descent algorithm applied to all datasets
    "step_size21": 2e-1,  # values of eta used for the modified gradient descent algorithm applied to a single dataset
    "step_size22": 1e-1,  # values of eta used for the modified gradient descent algorithm applied to all datasets
    "epsilon1": 0.00001,  # epsilon used for the normal gradient descent algorithm
    "epsilon2": 0.00001,  # epsilon used for the modified gradient descent algorithm
    "minimal_second_gradient": 0,   # value of m used for the modified gradient descent algorithm
    "e1": 0.01,  # value of e1 used in the calculating S^2
    "e2": 1/40,  # value of e2 used in the calculating S^2
    "single_data_set_num_steps": 25000,
    "all_data_sets_num_steps": 65000,
}

initial_parameters = {  # initial parameters for the gradient descent algorithms to start searching from
    "alpha": 2,
    "beta": 0.06,
    "gamma": 0.03,
    "delta": 0.2
}

# constants for the outputs
debug_mod = 1000  # number that indicates how much is printed in the console for debugging
model_graph_dt = 0.01

# create the model object
model = LV(dt=dt, max_t=max_t)

# do the analysis
analise(name, model, initial_parameters, data_file_names, analysis_parameters,
        debug_mod=debug_mod, model_graph_dt=model_graph_dt)
