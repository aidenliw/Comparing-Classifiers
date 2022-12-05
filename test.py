import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from main import import_data, categorize_data, separate_data
import time

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
X = np.concatenate((training_set_a_xy, training_set_b_xy))
y = np.concatenate((np.ones(len(training_set_a_xy)),
                    np.full(len(training_set_b_xy), fill_value=2, dtype=int)))


