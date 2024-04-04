import tensorflow as tf


class AutoEncoder:
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.en_decoder = None
        self.relu = tf.keras.activations.relu
        self.tanh = tf.keras.activations.tanh
        self.input_output_dim = 784
        self.encoder_hidden_layers = [200, 200]
        self.decoder_hidden_layers = [200, 200]
        self.code_dim = 32

    def build_model(self):
        ### Type your code here -- begin
        # Build Encoder
        # 입력층
        encoder_input = tf.keras.layers.Input(shape=(self.input_output_dim,), dtype=tf.float32)
        # 은닉층
        encoder_h_layer = encoder_input
        for dim in self.encoder_hidden_layers:
            encoder_h_layer = tf.keras.layers.Dense(
                units=dim, activation=self.relu, use_bias=True
            )(encoder_h_layer)
        # 출력층
        code = tf.keras.layers.Dense(
            units=self.code_dim, activation=self.tanh, use_bias=True
        )(encoder_h_layer)
        # 모델 생성
        self.encoder = tf.keras.models.Model(inputs=encoder_input, outputs=code)

        # Build Decoder
        # 입력층
        decoder_input = tf.keras.Input(shape=(self.code_dim,), dtype=tf.float32)
        # 은닉층
        decoder_h_layer = decoder_input
        for dim in self.decoder_hidden_layers:
            decoder_h_layer = tf.keras.layers.Dense(
                units=dim, activation=self.relu, use_bias=True
            )(decoder_h_layer)
        # 출력층
        decoder_output = tf.keras.layers.Dense(
            units=self.input_output_dim, activation=None, use_bias=True
        )(decoder_h_layer)
        # 모델 생성
        self.decoder = tf.keras.models.Model(inputs=decoder_input, outputs=decoder_output)

        # En-Decoder
        vae_output = self.decoder(code) # Encoder의 출력과 Decoder의 입력 연결
        self.en_decoder = tf.keras.models.Model(inputs=encoder_input, outputs=vae_output)
        optimizer_alg = tf.keras.optimizers.Adam(learning_rate=0.001)
        mse = tf.keras.losses.mse
        self.en_decoder.compile(optimizer=optimizer_alg, loss=mse)
        ### Type your code here -- end
        return

    def fit(self, x, y, batch_size, epochs):
        self.en_decoder.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)

    def save_weights(self, save_path):
        self.en_decoder.save_weights(save_path)

    def load_weights(self, load_path):
        self.en_decoder.load_weights(load_path)
