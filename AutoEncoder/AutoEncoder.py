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

        ### Type your code here -- end
        return

    def fit(self, x, y, batch_size, epochs):
        self.en_decoder.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)

    def save_weights(self, save_path):
        self.en_decoder.save_weights(save_path)

    def load_weights(self, load_path):
        self.en_decoder.load_weights(load_path)
