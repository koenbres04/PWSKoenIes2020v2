""""
- Koen Bresters 2020
Python file containing some small functions
"""


class DataException(Exception):
    """Raised when the provided data is invalid"""
    pass


# linearly interpolates between a and b
def lerp(a, b, t):
    return a*(1-t)+b*t


# get data from an .csv file
def get_data_from_csv(file_name, t_scale=1, x_scale=1, y_scale=1):
    data = []
    file = open(file_name)

    for line in file.readlines():
        line = line[:-1]
        line = line.replace(",", ".")
        i = line.find(";")
        j = line.find(";", i+1)
        data.append((float(line[:i])*t_scale, float(line[i+1:j])*x_scale, float(line[j+1:]))*y_scale)

    file.close()
    return data


# export an array of arrays to a .csv file
def export_array_array_to_csv(array_array, file_name, headers=None):
    lines = [";".join("" if item is None else str(item).replace(".", ",") for item in array) + "\n"
             for array in array_array]
    if headers is not None:
        lines.insert(0, ";".join(headers) + "\n")

    # write lines to the file
    file = open(file_name, "w")
    file.writelines(lines)
    file.close()


# export data to an .csv file
def export_data_to_svg(data, file_name, t_scale=1, x_scale=1, y_scale=1):
    # create each line of the file
    array_array = [[data_point[0]*t_scale, data_point[1]*x_scale, data_point[2]*y_scale] for data_point in data]

    # write lines to the file
    export_array_array_to_csv(array_array, file_name)


# finds initial conditions form an array of data-points
def find_t0_data_point(data):
    for data_point in data:
        if data_point[0] == 0:
            return data_point[1], data_point[2]
    raise DataException("Data contains no initial conditions!")


# finds data_point with largest time-value
def find_max_t_data_point(data):
    max_t_point = data[0]
    for data_point in data[1:]:
        if data_point[0] > max_t_point[0]:
            max_t_point = data_point
    return max_t_point


# a function that takes two time-dependent datasets
def unite_data(data1, data2):
    new_data = []
    a = len(data1[0])-1
    b = len(data2[0])-1
    i = 0
    j = 0
    while i < len(data1) and j < len(data2):
        if data1[i][0] > data2[j][0]:
            new_data.append([data2[j][0]] + [None for _ in range(a)] + [data2[j][k+1] for k in range(b)])
            j += 1
        elif data1[i][0] < data2[j][0]:
            new_data.append([data1[i][0]] + [data1[i][k+1] for k in range(a)] + [None for _ in range(b)])
            i += 1
        else:  # data1[i][0] == data2[j][0]
            new_data.append([data1[i][0]] + [data1[i][k+1] for k in range(a)] + [data2[j][k+1] for k in range(b)])
            i += 1
            j += 1
    # add remaining items from the dataset that isn't empty yet
    if i == len(data1):
        for n in range(j, len(data2)):
            new_data.append([data2[n][0]] + [None for _ in range(a)] + [data2[n][k+1] for k in range(b)])
    else:  # j == len(data2)
        for n in range(i, len(data1)):
            new_data.append([data2[n][0]] + [data1[n][k+1] for k in range(a)] + [None for _ in range(b)])
    return new_data


def conc(array):
    if not array:
        return []
    result = array[0]
    for x in array[1:]:
        result += x
    return result









