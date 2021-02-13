import time


# log results to a log file
def log_results(header, parameters, result_string, log_file):
    file = open(log_file, "a+")

    file.write("-"*10 + time.asctime() + "-"*10 + "\n")
    file.write("ran {} with parameters:\n".format(header))
    for key, value in parameters.items():
        file.write("{} = {}\n".format(key, value))
    file.write("\n")
    file.write(result_string)

    file.close()
