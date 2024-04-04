import tensorflow as tf
from MNISTData import MNISTData
from AutoEncoder import AutoEncoder
import numpy as np

if __name__ == "__main__":
    print("Hi. I am an AutoEncoder Tester.")

    data_loader = MNISTData()
    data_loader.load_data()
    ### Type your code here -- begin
    auto_encoder = AutoEncoder()
    auto_encoder.build_model()
    # 로드
    load_path = "./model/ae_model.weights.h5"
    auto_encoder.load_weights(load_path)
    print("load model weights from %s" % load_path)
    
    # 모델 평가 
    num_test_items = 56
    # 정답 이미지
    test_data = data_loader.x_test[: num_test_items, :]
    test_label = data_loader.y_test[: num_test_items]
    # Flatten 이미지를 2차원으로 복원
    test_data_x_print = test_data.reshape(num_test_items, data_loader.width, data_loader.height)

    print("const by codes")
    # 복원(예측) 이미지
    reconst_data = auto_encoder.en_decoder.predict(test_data)
    reconst_data_x_print = reconst_data.reshape(num_test_items, data_loader.width, data_loader.height)
    reconst_data_x_print = tf.math.sigmoid(reconst_data_x_print)
    # 정답과 복원 쌍 비교
    MNISTData.print_56_pair_images(test_data_x_print, reconst_data_x_print, test_label)

    print("const by code means for each digit")
    # 각 숫자마다 code의 평균을 계산해 출력: 공통 패턴 관찰
    avg_codes = np.zeros([10, 32])
    avg_add_cnt = np.zeros([10])
    
    latent_vecs = auto_encoder.encoder.predict(test_data)
    for i, label, in enumerate(test_label):
        avg_codes[label] = latent_vecs[i]
        avg_add_cnt[label] += 1.0
    for i in range(10):
        if avg_add_cnt[label] > 0.1:
            avg_codes[i] /= avg_add_cnt[label]

    avg_code_tensor = tf.convert_to_tensor(avg_codes)
    reconst_data_by_vecs = auto_encoder.decoder.predict(avg_code_tensor)
    reconst_data_x_by_mean_print = reconst_data_by_vecs.reshape(10, data_loader.width, data_loader.height)
    label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    MNISTData.print_10_images(reconst_data_x_by_mean_print, label_list)
    ### Type your code here -- end

