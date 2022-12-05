import numpy
import pandas


# Load data from the csv file by using pandas and convert it to numpy array
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
            class_a.append(item[indicator_index + 1:])
        elif indicator == 2:
            class_b.append(item[indicator_index + 1:])
        elif indicator == 3:
            class_c.append(item[indicator_index + 1:])
    return numpy.asarray(class_a), numpy.asarray(class_b), numpy.asarray(class_c)


# Separate the data by first 75% for training, and last 25% for testing
def separate_data(data):
    data_length = len(data)
    pivot = int(data_length * 0.75)
    training_set = data[:pivot]
    testing_set = data[pivot:]
    return training_set, testing_set


# Load data, Categorize data, then Separate data
def set_up_dataset():
    # Get data from imported csv file
    data = import_data()
    # categorize data by classes
    class_a, class_b, class_c = categorize_data(data)
    # separate data for training and testing sets
    training_set_a, testing_set_a = separate_data(class_a)
    training_set_b, testing_set_b = separate_data(class_b)
    # Get data with attribute x and y from the dataset for testing
    training_set_a_xy = training_set_a[:, 1:3]
    training_set_b_xy = training_set_b[:, 1:3]
    testing_set_a_xy = testing_set_a[:, 1:3]
    testing_set_b_xy = testing_set_b[:, 1:3]
    print("\n Loaded training set for class A with instances: " + str(len(training_set_a_xy)))
    print(" Loaded training set for class B with instances: " + str(len(training_set_b_xy)))
    print(" Loaded testing set for class A with instances: " + str(len(testing_set_a)))
    print(" Loaded testing set for class B with instances: " + str(len(testing_set_b)))
    return training_set_a_xy, training_set_b_xy, testing_set_a_xy, testing_set_b_xy