import matplotlib.pyplot as plt
import time
from FishersLinearDiscriminant import FishersLinearDiscriminant
from SupportVectorMachines import SupportVectorMachines
from DataLoader import set_up_dataset


# Execute the FLD methods
def run_fishers_linear_discriminant(training_set_a, training_set_b, testing_set_a, testing_set_b):

    # Instantiate the classes
    fld = FishersLinearDiscriminant()
    print("\n > Applying Fishers Linear Discriminant Classification")
    # Threshold trail and error
    # fld.fld_threshold_trail_and_error(training_set_a, training_set_b)

    # Train the data by using Fishers Linear Discriminant algorithm
    scaler = 8000000
    scaler_sk = 100
    threshold = -0.00000119
    start = time.time()
    Sw, w, slope, y_int = fld.train_fld_dataset(training_set_a, training_set_b, threshold)
    end = time.time()
    print(" > Computational times for training data is " +
          '{:.2f}'.format((end - start) * 1000) + " milliseconds")

    # Test the data
    start = time.time()
    true_positive, false_negative, true_negative, false_positive = \
        fld.test_fld_dataset(testing_set_a, testing_set_b, w, threshold)
    end = time.time()
    print(" > Computational times for testing data is " +
          '{:.2f}'.format((end - start) * 1000) + " milliseconds")
    print(" True Positive:  ", true_positive)
    print(" False Negative: ", false_negative)
    print(" True Negative:  ", true_negative)
    print(" False Positive: ", false_positive)

    # Plot the data
    fld.plot_fld_data(training_set_a, training_set_b, w, slope, y_int, scaler)
    plt.show()
    fld.plot_fld_data_with_error(training_set_a, training_set_b, w, slope, y_int, threshold, scaler)
    plt.show()
    # fld.plot_fld_data(testing_set_a, testing_set_b, w, slope, y_int, scaler)
    # plt.show()
    # fld.plot_fld_data_with_error(testing_set_a, testing_set_b, w, slope, y_int, threshold, scaler)
    # plt.show()

    # Sklearn implementation
    print(" \n > Applying Fishers Linear Discriminant Classification By Using Sklearn")
    # Train the data by using sklearn
    start = time.time()
    thresh_sk, w_sk, slope_sk, y_int_sk = fld.train_fld_dataset_sklearn(training_set_a, training_set_b)
    end = time.time()
    print(" > Computational Times for training data is "
          + '{:.2f}'.format((end - start) * 1000) + " milliseconds")

    # Test the data by using sklearn
    start = time.time()
    true_positive, false_negative, true_negative, false_positive = \
        fld.test_fld_dataset(testing_set_a, testing_set_b, w_sk, thresh_sk)
    end = time.time()
    print(" > Computational Times for Testing data is " +
          '{:.2f}'.format((end - start) * 1000) + " milliseconds")
    print(" True Positive:  ", true_positive)
    print(" False Negative: ", false_negative)
    print(" True Negative:  ", true_negative)
    print(" False Positive: ", false_positive)

    # Plot the data
    # fld.plot_fld_data(training_set_a, training_set_b, w_sk, slope_sk, y_int_sk, scaler_sk)
    # plt.show()
    # fld.plot_fld_data_with_error(training_set_a, training_set_b, w_sk, slope_sk, y_int_sk, threshold, scaler_sk)
    # plt.show()
    # fld.plot_fld_data(testing_set_a, testing_set_b, w_sk, slope_sk, y_int_sk, scaler_sk)
    # plt.show()
    # fld.plot_fld_data_with_error(testing_set_a, testing_set_b, w_sk, slope_sk, y_int_sk, threshold, scaler_sk)
    # plt.show()


# Execute the Support Vector Machines methods
def run_support_vector_machines(training_set_a, training_set_b, testing_set_a, testing_set_b):
    # Instantiate the classes
    svm = SupportVectorMachines()
    print(" \n > Applying Linear Support Vector Machines By Using Sklearn (It may takes several minutes...)")

    # Train the data
    start = time.time()
    cls = svm.train_svm_dataset_sklearn(training_set_a, training_set_b, "linear")
    end = time.time()
    print(" > Computational Times for training data is "
          + '{:.2f}'.format((end - start) * 1000) + " milliseconds")

    # Test the data
    start = time.time()
    true_positive, false_negative, true_negative, false_positive = \
        svm.test_svm_dataset_sklearn(testing_set_a, testing_set_b, cls)
    print(" True Positive:  ", true_positive)
    print(" False Negative: ", false_negative)
    print(" True Negative:  ", true_negative)
    print(" False Positive: ", false_positive)
    end = time.time()
    print(" > Computational Times for Testing data is " +
          '{:.2f}'.format((end - start) * 1000) + " milliseconds")

    # # Plot the Linear Support Vector Machines data. CAUTION! IT MAY TAKE MORE THAN 10 MINUTES
    # print("Plotting Linear Support Vector Machines data. It may take more than 10 minutes...")
    # svm.plot_linear_svm_dataset(training_set_a, training_set_b, "linear")

    print(" \n > Applying RBF Support Vector Machines By Using Sklearn (It may takes several minutes...)")

    # Train the data
    start = time.time()
    cls = svm.train_svm_dataset_sklearn(training_set_a, training_set_b, "rbf")
    end = time.time()
    print(" > Computational Times for training data is "
          + '{:.2f}'.format((end - start) * 1000) + " milliseconds")

    # Test the data
    start = time.time()
    true_positive, false_negative, true_negative, false_positive = \
        svm.test_svm_dataset_sklearn(testing_set_a, testing_set_b, cls)
    print(" True Positive:  ", true_positive)
    print(" False Negative: ", false_negative)
    print(" True Negative:  ", true_negative)
    print(" False Positive: ", false_positive)
    end = time.time()
    print(" > Computational Times for Testing data is " +
          '{:.2f}'.format((end - start) * 1000) + " milliseconds")

    # # Plot the RBF Support Vector Machines data. CAUTION! IT MAY TAKE MORE THAN 10 MINUTES
    # print("Plotting RBF Support Vector Machines data. It may take more than 10 minutes...")
    # svm.plot_linear_svm_dataset(training_set_a, training_set_b, "rbf")


# Retrieve data
training_set1, training_set2, testing_set1, testing_set2 = set_up_dataset()
run_fishers_linear_discriminant(training_set1, training_set2, testing_set1, testing_set2)
run_support_vector_machines(training_set1, training_set2, testing_set1, testing_set2)


