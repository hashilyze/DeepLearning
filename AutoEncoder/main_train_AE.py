from MNISTData import MNISTData
from AutoEncoder import AutoEncoder


if __name__ == "__main__":
    print("Hi. I am an Auto Encoder Trainer.")
    batch_size = 32
    num_epochs = 5

    data_loader = MNISTData()
    data_loader.load_data()

    ### Type your code here -- begin
    x_train = data_loader.x_train   # En-Decoder의 입력이자 정답
    input_output_dim = data_loader.in_out_dim

    auto_encoder = AutoEncoder()
    auto_encoder.build_model()
    auto_encoder.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=num_epochs)
    
    # 저장
    save_path = "./model/ae_model.weights.h5"
    auto_encoder.save_weights(save_path)
    print("save model weights to %s" % save_path)
    ### Type your code here -- end

