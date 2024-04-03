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
    print(train_labels)         # (60000, ): 각 이미지에 대한 분류 라벨(인덱스)
    print("Test data shape")
    print(test_images.shape)
    print("Test data labels")
    print(test_labels)

    ##### classifier train and predict – begin
    # 모델 생성
    # 28x28 크기의 이미지를 10개의 클래스로 분류
    image_classifier = ImageClassifier(28, 28, 10)
    image_classifier.build_MLP_model()
    # 훈련
    # 훈련 세트의 라벨(인덱스)을 모델의 출력 형태(10차원 벡터)로 변환 (원핫 인코딩)
    train_labels = ImageClassifier.to_onehotvec_label(train_labels, 10)
    image_classifier.fit(train_images, train_labels, num_epochs=100)
    # 예측
    predicted_labels = image_classifier.predict(test_images)
    # 모델의 예측(10차원 벡터)을 분류 라벨로 변환 (일치 확률이 가장 높은 클래스를 예측된 라벨로 결정)
    predicted_labels = tf.math.argmax(input=predicted_labels, axis=1) # axis = 0: 각 열마다 매핑, = 1: 각 행마다 매핑

    ## print predicted result
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[predicted_labels[i]])
    plt.show()
    ##### classifier train and predict – end

if __name__ == "__main__":
    # execute only if run as a script
    run_classifier()
