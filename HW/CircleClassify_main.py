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

    # ==== SLP =====
    mlp_classifier = MLP(hidden_layer_conf=[], num_output_nodes=1)
    mlp_classifier.build_model()
    mlp_classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)
    mlp_prediction = mlp_classifier.predict(x_test, batch_size=batch_size)
    mlp_prediction = tf.math.greater(mlp_prediction, tf.constant(0.5))

    plt.scatter(x_test[:, 0], x_test[:, 1], c=mlp_prediction, marker='.')
    plt.title("SLP prediction result")
    plt.show()

    # ===== MLP =====
    hidden_layer_confs = [
        [3, 3],
        [5, 5],
        [10, 10],
    ]
    for hidden_layer_conf in hidden_layer_confs:
        mlp_classifier = MLP(hidden_layer_conf=hidden_layer_conf, num_output_nodes=1)
        mlp_classifier.build_model()
        mlp_classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)
        mlp_prediction = mlp_classifier.predict(x_test, batch_size=batch_size)
        mlp_prediction = tf.math.greater(mlp_prediction, tf.constant(0.5))

        plt.scatter(x_test[:, 0], x_test[:, 1], c=mlp_prediction, marker='.')
        plt.title("MLP prediction result")
        plt.show()
    ############ Write your codes here - end

if __name__ == '__main__':
    CircleClassify()
