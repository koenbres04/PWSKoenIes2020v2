from ModelMain import gradient_descent_minimal_parameters, gradient_descent_minimal_parameters2, find_minimal_parameters
from CAModels import CAModel
from Utils import get_data_from_csv, export_array_array_to_csv, unite_data, find_t0_data_point, find_max_t_data_point
import math
import os

# file names
data_file_names = ["Data/exp4_beker1_data.csv",
                   "Data/exp4_beker2_data.csv",
                   "Data/exp4_beker3_data.csv",
                   "Data/exp4_beker4_data.csv"]
output_folder = "CA3DAnalysisResults"
single_data_set_errors_out = "single_data_set_errors.csv"
all_data_sets_errors_out = "all_data_sets_errors.csv"
single_data_set_prey_graph_format = "data_set_{}_prey_graph.csv"
single_data_set_predator_graph_format = "data_set_{}_predator_graph.csv"
all_data_sets_prey_graph_format = "all_data_data_set_{}_prey_graph.csv"
all_data_sets_predator_graph_format = "all_data_data_set_{}_predator_graph.csv"
results_out = "results.txt"

# constants
grid_shape = (64, 64, 64)
number_of_runs = 10  # value of L
max_t = 6.6
N = 10

epsilon1 = 0.00001  # epsilon and eta used for the normal gradient descent algorithm
step_size1 = 1e-7
epsilon2 = 0.00001  # epsilon and eta used for the modified gradient descent algorithm
step_size2 = 1e1
minimal_second_gradient = 1e-7  # value of m used for the modified gradient descent algorithm
y_factor = 100  # value of beta^2 in computing S^2 ; the value of alpha^2 is always kept at 1

single_data_set_num_steps = 10
all_data_sets_num_steps = 10

initial_parameters = {  # initial parameters for the gradient descent algorithms to start searching from
    "steps_per_time": 100,
    "x_factor": 100,
    "y_factor": 100,
    "sigma_f": 0.01,
    "p_f": 0.02,
    "mu": 0.03
}

# constants for the outputs
debug_mod = 1  # number that indicates how much is printed in the console for debugging
model_graph_dt = 0.01

# create an empty string for the contents of the result_file to be contained in
result_file_content = ""

# load data from files
datas = []
for file_name in data_file_names:
    datas.append(get_data_from_csv(file_name))


# create an approximation of a 3-dimensional version of Hawick, K. A., & Scogings, C. J. (2010) their model
model = CAModel(grid_shape, number_of_runs, max_t, N)

# create empty arrays for the errors to be stored in
single_data_errors = [[] for _ in range(2*len(datas))]
all_data_sets_errors = [[], []]

# create the output folder if it doesn't exist already:
if not os.path.exists(output_folder):
    os.mkdir(output_folder)


# run both the normal and modified gradient descent algorithms on all datasets seperately
for i in range(len(datas)):
    new_parameters1 = gradient_descent_minimal_parameters(datas[i], model, initial_parameters, epsilon1, step_size1,
                                                          single_data_set_num_steps,
                                                          y_factor=y_factor, error_out=single_data_errors[2*i],
                                                          debug_mod=debug_mod)
    new_parameters2 = gradient_descent_minimal_parameters2(datas[i], model, initial_parameters, epsilon2, step_size2,
                                                           single_data_set_num_steps,
                                                           y_factor=y_factor, error_out=single_data_errors[2*i+1],
                                                           debug_mod=debug_mod,
                                                           minimal_second_gradient=minimal_second_gradient)

    # write to the output file content string
    result_file_content += "Using data from {}:\n".format(data_file_names[i])
    result_file_content += "Normal gradient descent found the following parameters that gave error {}:\n {}\n".format(
        single_data_errors[2*i][-1], str(new_parameters1)
    )
    result_file_content += "Improved gradient descent found the following parameters that gave error {}:\n {}\n".format(
        single_data_errors[2*i + 1][-1], str(new_parameters2)
    )
    result_file_content += "\n"

    # write some files that are used to create graphs to visually compare the model
    x0, y0 = find_t0_data_point(datas[i])
    t_max = find_max_t_data_point(datas[i])[0]
    model_data1 = model.get_data(new_parameters1, x0, y0, 0, t_max, math.ceil(t_max/model_graph_dt+1))
    model_data2 = model.get_data(new_parameters2, x0, y0, 0, t_max, math.ceil(t_max/model_graph_dt+1))
    model_prey_data1 = [data_point[:2] for data_point in model_data1]
    model_prey_data2 = [data_point[:2] for data_point in model_data2]
    model_predator_data1 = [[data_point[0], data_point[2]] for data_point in model_data1]
    model_predator_data2 = [[data_point[0], data_point[2]] for data_point in model_data2]

    # write prey graph
    prey_graph_data = [data_point[:2] for data_point in datas[i]]
    prey_graph_data = unite_data(prey_graph_data, model_prey_data1)
    prey_graph_data = unite_data(prey_graph_data, model_prey_data2)
    export_array_array_to_csv(prey_graph_data, output_folder + "/" + single_data_set_prey_graph_format.format(i+1))

    # write predator graph
    predator_graph_data = [[data_point[0], data_point[2]] for data_point in datas[i]]
    predator_graph_data = unite_data(predator_graph_data, model_predator_data1)
    predator_graph_data = unite_data(predator_graph_data, model_predator_data2)
    export_array_array_to_csv(predator_graph_data,
                              output_folder + "/" + single_data_set_predator_graph_format.format(i+1))

# write the error graph file
single_data_errors_data = [[i] + [single_data_errors[j][i] for j in range(2*len(data_file_names))]
                           for i in range(single_data_set_num_steps+1)]
export_array_array_to_csv(single_data_errors_data, output_folder + "/" + single_data_set_errors_out)

# run both the normal and modified gradient descent algorithms on all datasets together
new_parameters1 = find_minimal_parameters(datas, model, initial_parameters, epsilon1, step_size1,
                                          all_data_sets_num_steps, use_improved_descent=False, y_factor=y_factor,
                                          error_out=all_data_sets_errors[0], debug_mod=debug_mod)
new_parameters2 = find_minimal_parameters(datas, model, initial_parameters, epsilon2, step_size2,
                                          all_data_sets_num_steps, use_improved_descent=True, y_factor=y_factor,
                                          error_out=all_data_sets_errors[1], debug_mod=debug_mod,
                                          minimal_second_gradient=minimal_second_gradient)
# write to the output file content string
result_file_content += "Using all data:\n"
result_file_content += "Normal gradient descent found the following parameters that gave error {}:\n {}\n".format(
    all_data_sets_errors[0][-1], str(new_parameters1)
)
result_file_content += "Improved gradient descent found the following parameters that gave error {}:\n {}\n".format(
    all_data_sets_errors[1][-1], str(new_parameters2)
)
result_file_content += "\n"

# write some files that are used to create graphs to visually compare the model
for i in range(len(datas)):
    x0, y0 = find_t0_data_point(datas[i])
    t_max = find_max_t_data_point(datas[i])[0]
    model_data1 = model.get_data(new_parameters1, x0, y0, 0, t_max, math.ceil(t_max/model_graph_dt+1))
    model_data2 = model.get_data(new_parameters2, x0, y0, 0, t_max, math.ceil(t_max/model_graph_dt+1))
    model_prey_data1 = [data_point[:2] for data_point in model_data1]
    model_prey_data2 = [data_point[:2] for data_point in model_data2]
    model_predator_data1 = [[data_point[0], data_point[2]] for data_point in model_data1]
    model_predator_data2 = [[data_point[0], data_point[2]] for data_point in model_data2]

    # write prey graph
    prey_graph_data = [data_point[:2] for data_point in datas[i]]
    prey_graph_data = unite_data(prey_graph_data, model_prey_data1)
    prey_graph_data = unite_data(prey_graph_data, model_prey_data2)
    export_array_array_to_csv(prey_graph_data, output_folder + "/" + all_data_sets_prey_graph_format.format(i+1))

    # write predator graph
    predator_graph_data = [[data_point[0], data_point[2]] for data_point in datas[i]]
    predator_graph_data = unite_data(predator_graph_data, model_predator_data1)
    predator_graph_data = unite_data(predator_graph_data, model_predator_data2)
    export_array_array_to_csv(predator_graph_data,
                              output_folder + "/" + all_data_sets_predator_graph_format.format(i+1))

# write the second error graph file
all_data_sets_errors_data = [[i] + [all_data_sets_errors[0][i], all_data_sets_errors[1][i]]
                             for i in range(all_data_sets_num_steps+1)]
export_array_array_to_csv(all_data_sets_errors_data, output_folder + "/" + all_data_sets_errors_out)

# write to the result file
result_file = open(output_folder + r"/" + results_out, "w")
result_file.write(result_file_content)
result_file.close()
