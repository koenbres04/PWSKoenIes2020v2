from SimpleModels import ExponentialPair
from AnalyisisMain import analise

# file names
data_file_names = ["Data/exp4_beker1_data.csv",
                   "Data/exp4_beker2_data.csv",
                   "Data/exp4_beker3_data.csv",
                   "Data/exp4_beker4_data.csv"]

name = "EP"

# constants
analysis_parameters = {
    "step_size11": 1e-2,  # values of eta used for the normal gradient descent algorithm applied to a single dataset
    "step_size12": 5e-2,  # values of eta used for the normal gradient descent algorithm applied to all datasets
    "step_size21": 1,  # values of eta used for the modified gradient descent algorithm applied to a single dataset
    "step_size22": 1,  # values of eta used for the modified gradient descent algorithm applied to all datasets
    "epsilon1": 1e-5,  # epsilon used for the normal gradient descent algorithm
    "epsilon2": 1e-5,  # epsilon used for the modified gradient descent algorithm
    "minimal_second_gradient": 0,   # value of m used for the modified gradient descent algorithm
    "e1": 0.01,  # value of e1 used in the calculating S^2
    "e2": 1/40,  # value of e2 used in the calculating S^2
    "single_data_set_num_steps": 5000,
    "all_data_sets_num_steps": 5000,
}

initial_parameters = {  # initial parameters for the gradient descent algorithms to start searching from
    "a": -1,
    "b": -1,
}

# constants for the outputs
debug_mod = 1000  # number that indicates how much is printed in the console for debugging
model_graph_dt = 0.01

# create the model object
model = ExponentialPair()

# do the analysis
analise(name, model, initial_parameters, data_file_names, analysis_parameters,
        debug_mod=debug_mod, model_graph_dt=model_graph_dt)
