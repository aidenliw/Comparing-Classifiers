import numpy
import pandas
import matplotlib.pyplot as plt
from FishersLinearDiscriminant import FishersLinearDiscriminant as fld


# Import data from the csv file by using pandas and convert it to numpy array
def import_data():

    # Import data from the Excel file by using pandas
    data = pandas.read_csv('accelerometer.csv', header=0)

    # Convert the dataset to a numpy array
    data_npy = numpy.asarray(data)

    return data_npy


# Separate the dataset by the class indicator value
def categorize_data(dataset):
    class_a = []
    class_b = []
    class_c = []

    # Get the index of the indicator, the indicator should be at the start of each line of the data
    indicator_index = 0
    for item in dataset:
        indicator = int(item[indicator_index])
        if indicator == 1:
            class_a.append(item[indicator_index+1:])
        elif indicator == 2:
            class_b.append(item[indicator_index+1:])
        elif indicator == 3:
            class_c.append(item[indicator_index+1:])
    return numpy.asarray(class_a), numpy.asarray(class_b), numpy.asarray(class_c)


# Separate the data by first 75% for training, and last 25% for testing
def separate_data(data):
    data_length = len(data)
    pivot = int(data_length * 0.75)
    training_set = data[:pivot]
    testing_set = data[pivot:]
    return training_set, testing_set


# Execute the methods
def run():
    # Get data from imported csv file
    data = import_data()
    # categorize data by classes
    class_a, class_b, class_c = categorize_data(data)
    # print(class_a)
    # separate data for training and testing sets
    training_set_a, testing_set_a = separate_data(class_a)
    training_set_b, testing_set_b = separate_data(class_b)
    # print(len(training_set_a))
    # print(len(training_set_b))
    training_set_a_xy = training_set_a[:, 1:3]
    training_set_b_xy = training_set_b[:, 1:3]
    # print(training_set_a_xy)
    # Sw, w, slope, y_int = calculate_fld(training_set_a_xy, training_set_b_xy, 0)
    scaler = 10000000
    threshold = -0.00000119
    Sw, w, slope, y_int = fld.calculate_fld(training_set_a_xy, training_set_b_xy, threshold)

    # print("Sw: ", Sw)
    # print("w: ", w)

    # for thresh in numpy.arange(-0.0000011, -0.0000013, -0.00000001):
    #     correct, error = fld.calculate_error(training_set_a_xy, training_set_b_xy, w, thresh)
    #     print("threshold = " + '{:.8f}'.format(round(thresh, 8))
    #           + "\t num of errors = " + str(error) + "\t Correct = " + str(correct))

    fld.plot_original_data(training_set_a_xy, training_set_b_xy, w, slope, y_int, scaler)
    plt.show()
    fld.plot_data_with_error(training_set_a_xy, training_set_b_xy, w, slope, y_int, threshold, scaler)
    plt.show()


run()
