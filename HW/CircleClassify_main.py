import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import tensorflow as tf
from MLP import MLP


def CircleClassify():
    # generating data
    n_samples = 400
    noise = 0.02
    factor = 0.5
    #### use x_train (Feature vectors), y_train (Class ground truths) as training set
    x_train, y_train = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    #### use x_test (Feature vectors) as test set
    #### you do not use y_test for this assignment.
    x_test, y_test = make_circles(n_samples=n_samples, noise=noise, factor=factor)

    #### visualizing training data distribution
    # plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker='.')
    # plt.title("Train data distribution")
    # plt.show()
    #
    # plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='.')
    # plt.title("Test data distribution")
    # plt.show()
    ############ Write your codes here - begin
    batch_size = 1
    epochs = 300

    hidden_layer_confs = [
        [],
        [3, 3],
        [5, 5],
        [10, 10],
    ]
    scores = []
    for hidden_layer_conf in hidden_layer_confs:
        classifier = MLP(hidden_layer_conf=hidden_layer_conf, num_output_nodes=1)
        classifier.build_model()
        classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)
        prediction = classifier.predict(x_test, batch_size=batch_size)
        prediction = tf.math.greater(prediction, tf.constant(0.5))
        scores.append(classifier.evaluate(x_test, y_test, batch_size=batch_size))

        plt.scatter(x_test[:, 0], x_test[:, 1], c=prediction, marker='.')
        if len(hidden_layer_conf) == 0:
            plt.title("SLP prediction result")
        else:
            plt.title("MLP prediction result")
        plt.show()

    for conf, acc in zip(hidden_layer_confs, scores):
        print("conf, acc: ", conf, acc)
    ############ Write your codes here - end

if __name__ == '__main__':
    CircleClassify()
