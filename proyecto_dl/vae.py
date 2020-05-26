"""VAE."""
from typing import Tuple
from functools import partial
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Sequential
from .data import Flickr8KImages, RoBERTaTokenizedFlickr8K
from tensorflow.keras import layers

from transformers import RobertaTokenizer, TFRobertaModel

class Sampling(layers.Layer):
    """Use mean and logvariance to sample normal encoding."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size,dim))
        return z_mean + tf.exp(0.5*z_log_var)*epsilon

        
class VAE(Model):

    def __init__(
        self,
        latent_dim: int,
        img_shape: Tuple[int, int],
        img_channels: int,
        **kwargs
    ):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.img_rows, self.img_cols = img_shape
        self.img_channels = img_channels
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)

    def train_step(self, data):
        image, text = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(text)
            reconstruction = self.decoder([z, text])
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(image, reconstruction)
            )
            reconstruction_loss *= 64*64
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = -0.5 * tf.reduce_mean(kl_loss)
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

class VAEBuilder:

    def return_(self):
        return self._vae

    def base_vae(
        self,
        latent_dim: int = 128,
        img_shape: Tuple[int, int] = (64, 64),
        img_channels: int = 3,
    ):
        self._vae = VAE(latent_dim, img_shape, img_channels)
        return self

    def build_roberta(
        self,
        model_name: str = "distilroberta-base",
        model_latent_dim: int = 768,
        max_sentence_length: int = 25,
    ):
        self._vae.roberta_shape = (max_sentence_length, model_latent_dim)
        roberta = TFRobertaModel.from_pretrained(model_name)
        # Se supone que esto evita pedos (https://github.com/huggingface/transformers/issues/1350#issuecomment-537625496)
        roberta.roberta.call = tf.function(roberta.roberta.call)
        self._vae.roberta = roberta
        return self

    def build_encoder(
        self,
        leaky_alpha: float = 0.2,
        kernel_size: int = 4,
        padding: str = "same",
        momentum: float = 0.8,
    ):

        # inputs
        sentence_length = self._vae.roberta_shape[0]
        text_params = layers.Input(
            shape=(sentence_length,), 
            dtype=tf.int64, 
            name="encoder_text_input"
        )
        roberta_embeddings = self._vae.roberta(text_params)[0]
        roberta_embeddings = layers.Bidirectional(layers.LSTM(64))(roberta_embeddings)


        # Texto a variables latentes
        params = layers.Dense(128)(roberta_embeddings)
        params = layers.LeakyReLU(alpha=leaky_alpha)(params)
        params = layers.Dense(128)(params)
        params = layers.LeakyReLU(alpha=leaky_alpha)(params)
        params = layers.Dense(64)(params)
        params = layers.LeakyReLU(alpha=leaky_alpha)(params)

        z_mean = layers.Dense(self._vae.latent_dim, name="z_mean")(params)
        z_log_var = layers.Dense(self._vae.latent_dim, name="z_log_var")(params)
        z = Sampling()([z_mean, z_log_var])
        encoder = Model(
            text_params,
            [z_mean, z_log_var, z],
            name="encoer" 
        )
        self._vae.encoder = encoder
        return self
    

    def build_decoder(self, leaky_alpha: float = 0.2):
        # inputs
        noise_input = layers.Input(shape=(self._vae.latent_dim,), name="decoder_noise_input")
        sentence_length = self._vae.roberta_shape[0]
        text_params = layers.Input(shape=(sentence_length,), dtype=tf.int64, name="decoder_text_input")
        roberta_embeddings = self._vae.roberta(text_params)[0]
        roberta_embeddings = layers.Bidirectional(layers.LSTM(64))(roberta_embeddings)

        adjusted_input = layers.multiply((roberta_embeddings, noise_input))

        # Generar imágenes
        params = layers.Dense(8*8*64, activation="relu")(adjusted_input)
        params = layers.Reshape((8, 8, 64))(params)
        params = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(params)
        params = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(params)
        params = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(params)
        outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(params)
        
        decoder = Model([noise_input, text_params], outputs, name="decoder")
        self._vae.decoder = decoder
        return self


class VAETrainer:

    LOG_DIR = "gs://tti-roberta-wcgan/tensorboard"
    MODEL_DIR = "gs://tti-roberta-wcgan/model-checkpoint"

    def __init__(
        self, 
        vae: VAE,
        log_dir: str = LOG_DIR, 
        model_dir = MODEL_DIR
    ):
        self.images = Flickr8KImages()
        self.text = RoBERTaTokenizedFlickr8K()
        self.histories = []
        self.log_dir = log_dir
        self.out_dir = model_dir
        self.vae = vae

    def _build_joint_data(self, batch_size: int):
        return (
            tf.data.Dataset
            .zip((self.images.test, self.text.test))
            .batch(batch_size)
        )

    def compile(self, optimizer: tf.keras.optimizers.Optimizer):
        self.vae.compile(optimizer)

    def train(self, epochs: int = 50, batch_size: int=36):
        data = self._build_joint_data(batch_size)
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.out_dir+"_{epoch}",
                save_freq="epoch"
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                update_freq="epoch"
            )
        ]
        history = self.vae.fit(data, epochs=epochs)#, callbacks=callbacks)
        self.histories.append(history)
        return history


    
