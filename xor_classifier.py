import tensorflow as tf
from MLP import MLP

def xor_classifier_example():
    # 입력 데이터
    input_data = tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    input_data = tf.cast(input_data, tf.float32)
    # 출력 데이터
    xor_labels = tf.constant([0.0, 1.0, 1.0, 0.0])
    xor_labels = tf.cast(xor_labels, tf.float32)
    # 하이퍼 파라미터
    batch_size = 1
    epochs = 1500

    # 입력층(2), 은닉층(4), 출력층(1)으로 구성
    mlp_classifier = MLP(hidden_layer_conf=[4], num_output_nodes=1)
    mlp_classifier.build_model()
    # 모델 훈련
    mlp_classifier.fit(x=input_data, y=xor_labels, batch_size=batch_size, epochs=epochs)
    
    # 테스트
    prediction = mlp_classifier.predict(input_data, batch_size=batch_size)
    input_and_result = zip(input_data, prediction)
    print("====== MLP XOR classifier result =====")
    for x, y in input_and_result:
        if y > 0.5:
            print("%d XOR %d => %.2f => 1" % (x[0], x[1], y))
        else:
            print("%d XOR %d => %.2f => 0" % (x[0], x[1], y))


if __name__ == '__main__':
    xor_classifier_example()

