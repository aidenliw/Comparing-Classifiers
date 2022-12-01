import numpy
import pandas
import matplotlib.pyplot as plt

numpy.set_printoptions(suppress=True)


class FishersLinearDiscriminant:

    # Calculate the data values of Sw, w, slope and y-intercept of the discriminant line
    # By using the fishers linear discriminant algorithm
    def calculate_fld(self, data_a, data_b, thresh):

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

        # Calculate the _w of the Linear Discriminant
        _w = numpy.dot(numpy.linalg.inv(_Sw), (mean_a - mean_b))

        # Calculate the slope and the y-intercept of the discriminant line
        _slope = -_w[0] / _w[1]
        y_intercept = thresh / _w[1]

        return _Sw, _w, _slope, y_intercept

    # Plot the data with original classes
    def plot_original_data(self, data_a, data_b, _w, _slope, _y_int, _scaler):

        # Plot the two dataset
        plt.scatter(x=data_a[:, 0], y=data_a[:, 1], c='red', marker='.')
        plt.scatter(x=data_b[:, 0], y=data_b[:, 1], c='blue', marker='.')

        # Plot the discriminant line
        axes = plt.gca()
        axes.set_aspect('equal', adjustable='box')
        x_vals = numpy.array(axes.get_xlim())
        y_vals = _y_int + _slope * x_vals
        plt.plot(x_vals, y_vals, 'g--', label='Discriminant Line')

        # Calculate and plot the line of the _w
        plt.plot([-_scaler * _w[0], _scaler * _w[0]], [-_scaler * _w[1], _scaler * _w[1]], 'y--', label='_w')
        plt.legend(loc='best')
        plt.title('Fishers Linear Discriminant')

    # Plot the data with the errors on the graph
    def plot_data_with_error(self, data_a, data_b, _w, _slope, _y_int, thresh, _scaler):
        self.plot_original_data(data_a, data_b, _w, _slope, _y_int, _scaler)

        # Generate category label based on the original dataset
        x_labels = numpy.concatenate((numpy.ones(len(data_a)),
                                      numpy.full(len(data_b), fill_value=2, dtype=numpy.int)))
        X = numpy.concatenate((data_a, data_b))

        # Get a list of predictions, if w^tx + Θ >= 0, return 1, elif w^tx + Θ < 0, return -1
        prediction = numpy.sign(numpy.dot(_w, X.T) + thresh)
        # Change all the -1 value to 2, for matching up the class value
        prediction[prediction < 0] = 2

        error = numpy.sum(prediction != x_labels)
        print("num errors = ", error)

        # Find the indices of array elements that are non-zero
        errorIndex = numpy.argwhere(prediction - x_labels)

        # Q = numpy.squeeze(X[errorIndex])
        # plt.scatter(Q[:, 0], Q[:, 1], c='g', marker='o')

        Q2 = X[errorIndex]
        plt.scatter(Q2[:, 0, 0], Q2[:, 0, 1], c='g', marker='.')
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

