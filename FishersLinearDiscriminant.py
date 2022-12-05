import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn import discriminant_analysis
numpy.set_printoptions(suppress=True)


class FishersLinearDiscriminant:

    # Train the given dataset by using the fishers linear discriminant algorithm
    # Return Sw, w, slope and y-intercept of the discriminant line
    def train_fld_dataset(self, data_a, data_b, thresh):

        # Calculate the mean of each dataset by using numpy.mean function
        mean_a = numpy.mean(data_a, axis=0)
        mean_b = numpy.mean(data_b, axis=0)
        # Subtract the mean from the data
        mean_a_centered = data_a - mean_a
        mean_b_centered = data_b - mean_b
        # Calculate the covariance of each dataset by using the equation (X^T*X) without dividing by n-1
        covariance_a = numpy.dot(mean_a_centered.T, mean_a_centered)
        covariance_b = numpy.dot(mean_b_centered.T, mean_b_centered)
        _Sw = covariance_a + covariance_b

        # Calculate the w of the Linear Discriminant
        _w = numpy.dot(numpy.linalg.inv(_Sw), (mean_a - mean_b))

        # Calculate the slope and the y-intercept of the discriminant line
        _slope = -_w[0] / _w[1]
        y_intercept = -thresh / _w[1]

        return _Sw, _w, _slope, y_intercept

    # Train the given dataset by using LinearDiscriminantAnalysis from sklearn library
    # Return slop and
    def train_fld_dataset_sklearn(self, data_a, data_b):
        X = numpy.concatenate((data_a, data_b))
        x_labels = numpy.concatenate((numpy.ones(len(data_a)),
                                      numpy.full(len(data_b), fill_value=2, dtype=numpy.int)))
        lda = discriminant_analysis.LinearDiscriminantAnalysis()
        lda.fit(X, x_labels)
        w = lda.coef_
        slope_sk = -lda.coef_[0][0] / lda.coef_[0][1]
        y_intercept = lda.intercept_
        thresh_sk1 = -y_intercept[0] / lda.coef_[0][1]
        thresh_sk = -lda.tol
        # covariance = lda.covariance_
        return thresh_sk, w[0], slope_sk, y_intercept[0]

    # Test the dataset by given dataset, w, and thresh value
    # Return true_positive, false_negative, true_negative, false_positive values
    def test_fld_dataset(self, data_a, data_b, _w, thresh):
        # Generate category label based on the original dataset
        x_labels = numpy.concatenate((numpy.ones(len(data_a)),
                                      numpy.full(len(data_b), fill_value=2, dtype=numpy.int)))

        X = numpy.concatenate((data_a, data_b))

        # Get a list of predictions, if w^tx + Θ >= 0, return 1, elif w^tx + Θ < 0, return -1
        prediction = numpy.sign(numpy.dot(_w, X.T) + thresh)
        # Change all the -1 value to 2, for matching up the class value
        prediction[prediction < 0] = 2

        # Get true_positive, false_negative, true_negative, false_positive from comparison
        true_positive = numpy.sum((prediction == x_labels) & (x_labels == 1))
        false_negative = numpy.sum((prediction != x_labels) & (x_labels == 1))
        true_negative = numpy.sum((prediction == x_labels) & (x_labels == 2))
        false_positive = numpy.sum((prediction != x_labels) & (x_labels == 2))

        return true_positive, false_negative, true_negative, false_positive

    # Plot the data with original classes
    def plot_fld_data(self, data_a, data_b, _w, _slope, _y_int, _scaler):

        # Plot the two dataset
        plt.scatter(x=data_a[:, 0], y=data_a[:, 1], c='red', marker='.', label='Class 1')
        plt.scatter(x=data_b[:, 0], y=data_b[:, 1], c='blue', marker='.', label='Class 2')

        # Plot the discriminant line
        axes = plt.gca()
        axes.set_aspect('equal', adjustable='box')
        x_vals = numpy.linspace(-3, 5, 100)
        # x_vals = numpy.array(axes.get_xlim())
        y_vals = _y_int + _slope * x_vals
        plt.plot(x_vals, y_vals, 'c--', label='Discriminant Line')

        # Calculate and plot the line of the _w
        plt.plot([-_scaler * _w[0], _scaler * _w[0]], [-_scaler * _w[1], _scaler * _w[1]], 'y--', label='W line')
        plt.legend(loc='best')
        plt.title('Fishers Linear Discriminant')

    # Plot the data with the errors on the graph
    def plot_fld_data_with_error(self, data_a, data_b, _w, _slope, _y_int, thresh, _scaler):
        self.plot_fld_data(data_a, data_b, _w, _slope, _y_int, _scaler)

        # Generate category label based on the original dataset
        x_labels = numpy.concatenate((numpy.ones(len(data_a)),
                                      numpy.full(len(data_b), fill_value=2, dtype=numpy.int)))
        X = numpy.concatenate((data_a, data_b))

        # Get a list of predictions, if w^tx + Θ >= 0, return 1, elif w^tx + Θ < 0, return -1
        prediction = numpy.sign(numpy.dot(_w, X.T) + thresh)
        # Change all the -1 value to 2, for matching up the class value
        prediction[prediction < 0] = 2

        # Find the indices of array elements that are non-zero
        errorIndex = numpy.argwhere(prediction - x_labels)

        Q2 = X[errorIndex]
        plt.scatter(Q2[:, 0, 0], Q2[:, 0, 1], c='g', marker='.', label="Errors")
        plt.legend(loc='best')
        plt.title('Fishers Linear Discriminant With Errors')

    # Calculate the error numbers by given dataset, w, and thresh value
    def calculate_error(self, data_a, data_b, _w, thresh):
        # Generate category label based on the original dataset
        x_labels = numpy.concatenate((numpy.ones(len(data_a)),
                                      numpy.full(len(data_b), fill_value=2, dtype=numpy.int)))

        X = numpy.concatenate((data_a, data_b))

        # Get a list of predictions, if w^tx + Θ >= 0, return 1, elif w^tx + Θ < 0, return -1
        prediction = numpy.sign(numpy.dot(_w, X.T) + thresh)
        # Change all the -1 value to 2, for matching up the class value
        prediction[prediction < 0] = 2

        correct = numpy.sum(prediction == x_labels)
        error = numpy.sum(prediction != x_labels)

        return correct, error

    # Find the best case scenario of the threshold for the lieaner discriminant
    def fld_threshold_trail_and_error(self, data_a, data_b):
        print(" > Threshold Trail and error")
        Sw, w, slope, y_int = self.train_fld_dataset(data_a, data_b, 0)
        for thresh in numpy.arange(0.000005, -0.000005, -0.00000001):
            correct, error = self.calculate_error(data_a, data_b, w, thresh)
            print(" threshold = " + '{:.8f}'.format(round(thresh, 8))
                  + "\t num of errors = " + str(error) + "\t Correct = " + str(correct))


