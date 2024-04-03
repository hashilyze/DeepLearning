import tensorflow as tf
import matplotlib.pyplot as plt
from ImageClassifier import ImageClassifier
import numpy as np


def run_classifier():
    # fashion mnist Dataset을 가져옴
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # 분류할 classes의 설명(이름)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print("Train data shape")
    print(train_images.shape)   # (60000, 28, 28): 28x28 크기의 흑백 이미지 6만 장
    print("Train data labels")
    print(train_labels)
    print("Test data shape")
    print(test_images.shape)
    print("Test data labels")
    print(test_labels)

    ##### classifier train and predict – begin



    '''
    ## print predicted result
    predicted_labels = None    
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[predicted_labels[i]])
    plt.show()
    '''
    ##### classifier train and predict – end

if __name__ == "__main__":
    # execute only if run as a script
    run_classifier()