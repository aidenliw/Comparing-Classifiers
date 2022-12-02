import numpy
import pandas
import matplotlib.pyplot as plt
import time
from FishersLinearDiscriminant import FishersLinearDiscriminant


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


# Execute the methods
def run():
    print(" > Applying Fishers Linear Discriminant Classification")
    # Instantiate the classes
    fld = FishersLinearDiscriminant()
    # Get data from imported csv file
    data = import_data()
    # categorize data by classes
    class_a, class_b, class_c = categorize_data(data)
    # separate data for training and testing sets
    training_set_a, testing_set_a = separate_data(class_a)
    training_set_b, testing_set_b = separate_data(class_b)
    # Get attribute x and y from the dataset for testing
    training_set_a_xy = training_set_a[:, 1:3]
    training_set_b_xy = training_set_b[:, 1:3]
    testing_set_a_xy = testing_set_a[:, 1:3]
    testing_set_b_xy = testing_set_b[:, 1:3]

    # # Find the best case scenario of the threshold for the lieaner discriminant
    # for thresh in numpy.arange(-0.0000011, -0.0000013, -0.00000001):
    #     correct, error = fld.calculate_error(training_set_a_xy, training_set_b_xy, w, thresh)
    #     print("threshold = " + '{:.8f}'.format(round(thresh, 8))
    #           + "\t num of errors = " + str(error) + "\t Correct = " + str(correct))

    # Train the data by using Fishers Linear Discriminant algorithm
    scaler = 8000000
    scaler_sk = 100
    threshold = -0.00000119
    start = time.time()
    Sw, w, slope, y_int = fld.train_fld_dataset(training_set_a_xy, training_set_b_xy, threshold)
    end = time.time()
    print(" > Computational Times for training data is " +
          '{:.2f}'.format((end - start) * 1000) + " milliseconds")

    # Train the data by using Fishers Linear Discriminant algorithm from sklearn
    start = time.time()
    thresh_sk, w_sk, slope_sk, y_int_sk = fld.train_fld_dataset_sklearn(training_set_a_xy, training_set_b_xy)
    end = time.time()
    print(" > Computational Times for training data by using sklearn is "
          + '{:.2f}'.format((end - start) * 1000) + " milliseconds")

    # Test the data
    start = time.time()
    true_positive, false_negative, true_negative, false_positive = \
        fld.test_fld_dataset(testing_set_a_xy, testing_set_b_xy, w, threshold)
    end = time.time()
    print(" > Computational Times for Testing data is " +
          '{:.2f}'.format((end - start) * 1000) + " milliseconds")

    print(" True Positive:  ", true_positive)
    print(" False Negative: ", false_negative)
    print(" True Negative:  ", true_negative)
    print(" False Positive: ", false_positive)

    # Plot the data
    fld.plot_original_data(training_set_a_xy, training_set_b_xy, w, slope, y_int, scaler)
    plt.show()
    fld.plot_data_with_error(training_set_a_xy, training_set_b_xy, w, slope, y_int, threshold, scaler)
    plt.show()

    # # Plot the data
    # fld.plot_original_data(training_set_a_xy, training_set_b_xy, w_sk, slope_sk, y_int_sk, scaler_sk)
    # plt.show()
    # fld.plot_data_with_error(training_set_a_xy, training_set_b_xy, w_sk, slope_sk, y_int_sk, threshold, scaler_sk)
    # plt.show()


run()
