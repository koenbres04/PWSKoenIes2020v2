import os
from Utils import get_data_from_csv, conc
from QuickDataVisualizer import write_comparison_csv, export_array_array_to_csv
from ModelMain import find_minimal_parameters
from Logging import log_results
import pandas as pd
import ntpath

single_data_set_errors_out = "single_data_set_errors.csv"
all_data_sets_errors_out = "all_data_sets_errors.csv"
single_data_set_parameters_out = "single_data_set_parameters.csv"
all_data_sets_parameters_out = "all_data_sets_parameters.csv"
single_data_set_graph_format = "data_set_{}_graph.csv"
all_data_sets_graph_format = "all_data_data_set_{}_graph.csv"
resulting_data_excel_format = "{}_resulting_data.xlsx"
results_out = "results.txt"


def analise(name, model, initial_parameters, data_file_names, analysis_parameters, debug_mod=1000, model_graph_dt=0.01,
            output_folder_location=""):
    # some definitions
    step_size11 = analysis_parameters["step_size11"]
    step_size12 = analysis_parameters["step_size11"]
    step_size21 = analysis_parameters["step_size21"]
    step_size22 = analysis_parameters["step_size22"]
    epsilon1 = analysis_parameters["epsilon1"]
    epsilon2 = analysis_parameters["epsilon2"]
    minimal_second_gradient = analysis_parameters["minimal_second_gradient"]
    e1 = analysis_parameters["e1"]
    e2 = analysis_parameters["e2"]
    single_data_set_num_steps = analysis_parameters["single_data_set_num_steps"]
    all_data_sets_num_steps = analysis_parameters["all_data_sets_num_steps"]

    output_folder = name + "AnalysisResults" if output_folder_location == "" else \
        output_folder_location + os.path.sep + name + "AnalysisResults"
    log_file = name + "_log.txt"

    # create an empty string for the contents of the result_file to be contained in
    result_file_content = ""

    # create an empty array to store the names of all of the generated csv files
    csv_file_names = []

    # load data from files
    datas = []
    for file_name in data_file_names:
        datas.append(get_data_from_csv(file_name))

    # create empty arrays for the errors and parameters to be stored in
    single_data_errors = [[] for _ in range(2*len(datas))]
    all_data_sets_errors = [[], []]
    single_data_parameters = [[] for _ in range(2*len(datas))]
    all_data_parameters = [[], []]

    # create the output folder if it doesn't exist already:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # run both the normal and modified gradient descent algorithms on all datasets seperately
    for i in range(len(datas)):
        new_parameters1 = find_minimal_parameters([datas[i]], model, initial_parameters, epsilon1, step_size11,
                                                  single_data_set_num_steps, use_improved_descent=False,
                                                  error_out=single_data_errors[2*i],
                                                  parameter_out=single_data_parameters[2*i],
                                                  debug_mod=debug_mod, e1=e1, e2=e2)

        new_parameters2 = find_minimal_parameters([datas[i]], model, initial_parameters, epsilon2, step_size21,
                                                  single_data_set_num_steps, use_improved_descent=True,
                                                  error_out=single_data_errors[2*i+1],
                                                  parameter_out=single_data_parameters[2*i+1],
                                                  minimal_second_gradient=minimal_second_gradient,
                                                  debug_mod=debug_mod, e1=e1, e2=e2)

        # write to the output file content string
        result_file_content += "Using data from {}:\n".format(data_file_names[i])
        result_file_content += "Normal gradient descent found the following parameters that gave error {}:\n {}\n".format(
            min(single_data_errors[2*i]), str(new_parameters1)
        )
        result_file_content += "Improved gradient descent found the following parameters that gave error {}:\n {}\n".format(
            min(single_data_errors[2*i + 1]), str(new_parameters2)
        )
        result_file_content += "\n"

        # write some files that are used to create graphs to visually compare the model
        csv_file_name = output_folder + "/" + single_data_set_graph_format.format(i+1)
        write_comparison_csv(csv_file_name, datas[i],
                             model, [new_parameters1, new_parameters2], model_dt=model_graph_dt,
                             headers=["D" + str(i+1), "{} {} {}".format(name, "D" + str(i+1), "NGD"),
                                      "{} {} {}".format(name, "D" + str(i+1), "AGD")])
        csv_file_names.append(csv_file_name)

    # write the error graph file
    single_data_errors_data = [[i0] + [single_data_errors[j][i0] for j in range(2*len(datas))]
                               for i0 in range(single_data_set_num_steps+1)]
    csv_file_name = output_folder + "/" + single_data_set_errors_out
    headers = [""]
    for i in range(len(datas)):
        headers.append("{} D{} NGD".format(name, str(i+1)))
        headers.append("{} D{} AGD".format(name, str(i+1)))
    export_array_array_to_csv(single_data_errors_data, csv_file_name, headers=headers)
    csv_file_names.append(csv_file_name)

    # write the parameters file
    single_data_parameters_data = [[None] + conc([["{}_{}_{}".format(str(k//2), str(k%2+1), key)
                                   for key in initial_parameters.keys()] for k in range(2*len(datas))])]
    single_data_parameters_data += [[i0] + conc([single_data_parameters[j][i0] for j in range(2*len(datas))])
                                    for i0 in range(single_data_set_num_steps+1)]
    csv_file_name = output_folder + "/" + single_data_set_parameters_out
    export_array_array_to_csv(single_data_parameters_data, csv_file_name)
    del single_data_parameters_data
    del single_data_parameters

    # run both the normal and modified gradient descent algorithms on all datasets together
    new_parameters1 = find_minimal_parameters(datas, model, initial_parameters, epsilon1, step_size12,
                                              all_data_sets_num_steps, use_improved_descent=False,
                                              parameter_out=all_data_parameters[0],
                                              error_out=all_data_sets_errors[0], debug_mod=debug_mod, e1=e1, e2=e2)
    new_parameters2 = find_minimal_parameters(datas, model, initial_parameters, epsilon2, step_size22,
                                              all_data_sets_num_steps, use_improved_descent=True,
                                              parameter_out=all_data_parameters[1],
                                              error_out=all_data_sets_errors[1], debug_mod=debug_mod,
                                              minimal_second_gradient=minimal_second_gradient, e1=e1, e2=e2)
    # write to the output file content string
    result_file_content += "Using all data:\n"
    result_file_content += "Normal gradient descent found the following parameters that gave error {}:\n {}\n".format(
        min(all_data_sets_errors[0]), str(new_parameters1)
    )
    result_file_content += "Improved gradient descent found the following parameters that gave error {}:\n {}\n".format(
        min(all_data_sets_errors[1]), str(new_parameters2)
    )
    result_file_content += "\n"

    # write some files that are used to create graphs to visually compare the model
    for i in range(len(datas)):
        csv_file_name = output_folder + "/" + all_data_sets_graph_format.format(i + 1)
        write_comparison_csv(csv_file_name, datas[i],
                             model, [new_parameters1, new_parameters2], model_dt=model_graph_dt,
                             headers=["D" + str(i+1), "{} {} {}".format(name, "AD", "NGD"),
                                      "{} {} {}".format(name, "D" + str(i+1), "AGD")])
        csv_file_names.append(csv_file_name)

    # write the second error graph file
    all_data_sets_errors_data = [[i] + [all_data_sets_errors[0][i], all_data_sets_errors[1][i]]
                                 for i in range(all_data_sets_num_steps+1)]
    csv_file_name = output_folder + "/" + all_data_sets_errors_out
    export_array_array_to_csv(all_data_sets_errors_data, csv_file_name,
                              headers=["", name + " AD NGD", name + " AD AGD"])
    csv_file_names.append(csv_file_name)

    # write the second parameters file
    all_data_parameters_data = [[None] + conc([["{}_{}".format(str(k), key)
                                                for key in initial_parameters.keys()] for k in range(2)])]
    all_data_parameters_data += [[i0] + all_data_parameters[0][i0] + all_data_parameters[1][i0]
                                 for i0 in range(all_data_sets_num_steps + 1)]
    csv_file_name = output_folder + "/" + all_data_sets_parameters_out
    export_array_array_to_csv(all_data_parameters_data, csv_file_name)
    del all_data_parameters_data
    del all_data_parameters

    # write to the result excel file
    excel_writer = pd.ExcelWriter(output_folder + os.path.sep + resulting_data_excel_format.format(name))
    for file_name in csv_file_names:
        df = pd.read_csv(file_name, sep=";", decimal=",", header=0)
        df.to_excel(excel_writer, os.path.splitext(ntpath.basename(file_name))[0], header=True, index=False)
    excel_writer.save()

    # write to the result file
    result_file = open(output_folder + r"/" + results_out, "w")
    result_file.write(result_file_content)
    result_file.close()

    # write to log file
    log_results(name, analysis_parameters, result_file_content, log_file)


def fetch_parameters(folder: str, all_datasets: bool, step_num: int, method="both", data_set_num=1):
    csv_file_name = folder + os.path.sep + (all_data_sets_parameters_out
                                            if all_datasets else single_data_set_parameters_out)
    data_set_num -= 1
    header = ""
    text = ""
    file = open(csv_file_name)
    for i, line in enumerate(file):
        if i == 0:
            header = line
        if i == step_num+1:
            text = line
            break
    file.close()

    headers = header[:-1].split(";")[1:]
    if all_datasets:
        keys = [t[t.find("_")+1:] for t in headers]
    else:
        keys = [t[t.find("_", t.find("_")+1)+1:] for t in headers]
    new_keys = []
    for key in keys:
        if key not in new_keys:
            new_keys.append(key)
    keys = new_keys
    num_params = len(keys)

    items = text.replace(",", ".")[:-1].split(";")[1:]
    if method != "both":
        n = 0 if method == "NGD" else 1
        if all_datasets:
            values = [float(t) for t in items[n*num_params:(n+1)*num_params]]
        else:
            values = [float(t) for t in items[(data_set_num*2+n)*num_params:(data_set_num*2+n+1)*num_params]]
        return dict(zip(keys, values))
    else:
        values = []
        for n in range(2):
            if all_datasets:
                values.append([float(t) for t in items[n * num_params:(n + 1) * num_params]])
            else:
                values.append([float(t) for t in
                               items[(data_set_num * 2 + n) * num_params:(data_set_num * 2 + n + 1) * num_params]])
        return tuple(dict(zip(keys, v)) for v in values)
