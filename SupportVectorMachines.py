import numpy
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
numpy.set_printoptions(suppress=True)


class SupportVectorMachines:

    # Train the given dataset by using Support Vector Machines from sklearn library
    # Return the classifier
    def train_svm_dataset_sklearn(self, data_a, data_b, kernel):
        X = numpy.concatenate((data_a, data_b))
        x_labels = numpy.concatenate((numpy.ones(len(data_a)),
                                      numpy.full(len(data_b), fill_value=2, dtype=numpy.int)))
        classifier = svm.SVC(kernel=kernel)
        classifier.fit(X, x_labels)

        return classifier

    # Test the given dataset by using Support Vector Machines from sklearn library
    def test_svm_dataset_sklearn(self, data_a, data_b, classifier):
        # Generate category label based on the original dataset
        x_labels = numpy.concatenate((numpy.ones(len(data_a)),
                                      numpy.full(len(data_b), fill_value=2, dtype=numpy.int)))

        X = numpy.concatenate((data_a, data_b))

        # Get a list of predictions
        prediction = classifier.predict(X)

        # Get true_positive, false_negative, true_negative, false_positive from comparison
        true_positive = numpy.sum((prediction == x_labels) & (x_labels == 1))
        false_negative = numpy.sum((prediction != x_labels) & (x_labels == 1))
        true_negative = numpy.sum((prediction == x_labels) & (x_labels == 2))
        false_positive = numpy.sum((prediction != x_labels) & (x_labels == 2))

        return true_positive, false_negative, true_negative, false_positive

    # Plot the data by given kernel type ('linear' or 'rbf')
    def plot_linear_svm_dataset(self, data_a, data_b, kernel):
        X = numpy.concatenate((data_a, data_b))
        y = numpy.concatenate((numpy.ones(len(data_a)),
                               numpy.full(len(data_b), fill_value=2, dtype=int)))

        svc = svm.SVC(kernel='kernel')
        svc.fit(X, y)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # Put the result into a color plot
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        # Plot the decision boundary. For that, we will assign a color to each
        DecisionBoundaryDisplay.from_estimator(svc, X, cmap=cm, alpha=0.8, eps=0.5)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors="k")
        plt.xlabel('Sepal Length')
        plt.ylabel('Sepal Width')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        cap_name = kernel.capitalize()
        plt.title('SVC with ' + cap_name + ' Kernel')
        plt.show()

