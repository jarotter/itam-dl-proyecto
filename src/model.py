"""GAN."""

from typing import Tuple

from functools import partial
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    BatchNormalization,
    Bidirectional,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    Reshape,
    UpSampling2D,
    ZeroPadding2D,
    concatenate,
    multiply,
)

from transformers import RobertaTokenizer, TFRobertaModel


class RandomWeightedAverage(tf.keras.layers.Layer):
    """Recta entre dos imágenes."""

    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        alpha = tf.random.uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class WCGAN(Model):
    """Wasserstein GAN condicional con RoBERTa."""

    def __init__(
        self,
        latent_dim: int,
        img_shape: Tuple[int, int],
        img_channels: int,
        discriminator_loops: int = 5,
    ):
        super(WCGAN, self).__init__()
        self.latent_dim = latent_dim
        self.img_rows, self.img_cols = img_shape
        self.img_channels = img_channels
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.discriminator_loops = discriminator_loops

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def gradient_penalty_loss(self, y_pred, averaged_samples):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(
            gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape))
        )
        gradient_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = K.square(1 - gradient_norm)
        return K.mean(gradient_penalty)

    def compile(self, discriminator_optimizer, generator_optimizer, loss):
        super(WCGAN, self).compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.loss_fn = loss

    def train_step(self, real_images, preprocessed_texts):
        # El tamaño del batch
        assert tf.shape(real_images)[0] == tf.shape(preprocessed_texts)[0]
        batch_size = tf.shape(real_images[0])

        # Etiquetas
        real_labels = -np.ones((batch_size, 1))
        np.ones((batch_size, 1))

        for _ in range(self.discriminator_loops):
            # Generar imágenes
            random_noise = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_images = self.generator([random_noise, preprocessed_texts])

            # Interpolar imágenes
            interpolated_images = RandomWeightedAverage()(
                [real_images, generated_images]
            )

            # Entrenar al discriminador
            with tf.GradientTape() as tape:
                real_pred = self.discriminator([real_images, preprocessed_texts])
                real_loss = self.wasserstein_loss(real_labels, real_pred)
                generated_pred = self.discriminator(
                    [generated_images, preprocessed_texts]
                )
                generated_loss = self.wasserstein_loss(generated_labels, generated_pred)
                interpolated_pred = self.discriminator(
                    [interpolated_images, preprocessed_texts]
                )
                interpolated_loss = self.gradient_penalty_loss(
                    interpolated_pred, interpolated_images
                )
                discriminator_loss = real_loss + generated_loss + 10 * interpolated_loss
            grads = tape.gradient(
                discriminator_loss, self.discriminator.trainable_weights
            )
            self.discriminator_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )

        # Entrenar al generador
        with tf.GradientTape() as tape:
            generated_images = self.generator([random_noise, preprocessed_texts])
            generated_pred = self.discriminator(generated_images)
            generator_loss = self.wasserstein_loss(real_labels, generated_pred)
        grads = tape.gradient(generator_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights)
        )


class WCGANBuilder:
    """
    Aplica el patrón constructor para facilitar la 
    parametrización de WCGAN.
    """

    def base_wcgan(
        self,
        latent_dim: int = 128,
        img_shape: Tuple[int, int] = (64, 64),
        img_channels: int = 3,
    ):
        self._model = WCGAN(latent_dim, img_shape, img_channels)
        return self

    def build_roberta(
        self,
        model_name: str = "distilroberta-base",
        model_latent_dim: int = 768,
        max_sentence_length: int = 25,
    ):
        self._model.roberta_shape = (max_sentence_length, model_latent_dim)
        roberta = TFRobertaModel.from_pretrained(model_name)
        # Se supone que esto evita pedos (https://github.com/huggingface/transformers/issues/1350#issuecomment-537625496)
        roberta.roberta.call = tf.function(roberta.roberta.call)
        self._model.roberta = roberta
        return self

    def build_discriminator(
        self,
        padding: str = "same",
        leaky_alpha: float = 0.2,
        dropout_probability: float = 0.25,
        kernel_size: int = 3,
    ):

        img_input = Input(shape=self._model.img_shape)
        img_params = Conv2D(
            filters=16,
            kernel_size=kernel_size,
            strides=2,
            input_shape=self._model.img_shape,
            padding=padding,
        )(img_input)
        img_params = LeakyReLU(alpha=leaky_alpha)(img_params)
        img_params = Dropout(dropout_probability)(img_params)
        img_params = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=2,
            input_shape=self._model.img_shape,
            padding=padding,
        )(img_params)
        img_params = LeakyReLU(alpha=leaky_alpha)(img_params)
        img_params = Dropout(dropout_probability)(img_params)
        img_params = Conv2D(
            filters=64, kernel_size=kernel_size, strides=2, padding=padding
        )(img_params)
        img_params = LeakyReLU(alpha=leaky_alpha)(img_params)
        img_params = Dropout(dropout_probability)(img_params)
        img_params = Conv2D(
            filters=128, kernel_size=kernel_size, strides=2, padding=padding
        )(img_params)
        img_params = ZeroPadding2D(padding=((0, 1), (0, 1)))(img_params)
        img_params = LeakyReLU(alpha=leaky_alpha)(img_params)
        img_params = Dropout(dropout_probability)(img_params)
        img_params = Reshape((128 * 5 * 5,))(img_params)

        # Aquí voy a sustituir lo que hacen de las labels
        # para utilizar bert
        sentence_length = self._model.roberta_shape[0]
        text_params = Input(shape=(sentence_length,), dtype=tf.int64)
        roberta_embeddings = self._model.roberta(text_params)[0]
        roberta_embeddings = Dense(128)(roberta_embeddings)
        roberta_embeddings = Reshape((128 * 5 * 5,))(roberta_embeddings)

        joint_params = concatenate((roberta_embeddings, img_params))
        joint_params = Dense(512)(joint_params)
        joint_params = LeakyReLU(alpha=leaky_alpha)(joint_params)
        joint_params = Dense(128)(joint_params)
        joint_params = LeakyReLU(alpha=leaky_alpha)(joint_params)
        joint_params = Dense(32)(joint_params)
        joint_params = LeakyReLU(alpha=leaky_alpha)(joint_params)
        output = Dense(1)(joint_params)

        model = Model([img_input, text_params], output)
        self._model.discriminator = model
        return self

    def build_generator(
        self,
        leaky_alpha: float = 0.2,
        kernel_size: int = 4,
        padding: str = "same",
        momentum: float = 0.8,
    ):

        # inputs
        noise_input = Input(shape=(self._model.latent_dim,))
        sentence_length = self._model.roberta_shape[0]
        text_params = Input(shape=(sentence_length,), dtype=tf.int64)
        roberta_embeddings = self._model.roberta(text_params)[0]
        roberta_embeddings = Bidirectional(LSTM(64))(roberta_embeddings)

        adjusted_input = multiply((roberta_embeddings, noise_input))

        # Construir imagen
        params = Dense(128 * 8 * 8)(adjusted_input)
        params = LeakyReLU(alpha=leaky_alpha)(params)
        params = Reshape((8, 8, 128))(params)
        params = UpSampling2D()(params)
        params = Conv2D(128, kernel_size=kernel_size, padding=padding)(params)
        params = BatchNormalization(momentum=momentum)(params)
        params = LeakyReLU(alpha=leaky_alpha)(params)
        params = UpSampling2D()(params)
        params = Conv2D(64, kernel_size=kernel_size, padding=padding)(params)
        params = BatchNormalization(momentum=momentum)(params)
        params = LeakyReLU(alpha=leaky_alpha)(params)
        params = UpSampling2D()(params)
        params = Conv2D(
            self._model.img_channels, kernel_size=kernel_size, padding="same"
        )(params)
        img = Activation("tanh")(params)

        model = Model([noise_input, text_params], img)
        self._model.generator = model
        return self


class GANAugmenter:

    def __init__(self, gan: WCGAN):
        self.gan = gan

    def build_discriminator_model(self):
        self.gan.generator.trainable = False

        # Imágenes reales
        real_img = Input(shape=self.gan.img_shape)

        # Imágenes generadas
        noise_input = Input(shape=(self.gan.latent_dim,))
        text_input = Input(shape=(self.gan.roberta_shape[0],))
        fake_img = self.gan.generator([noise_input, text_input])

        # Discriminador
        fake_pred = self.gan.discriminator([fake_img, text_input])
        real_pred = self.gan.discriminator([real_img, text_input])

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        interpolated_pred = self.gan.discriminator([interpolated_img, text_input])

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gan.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names
        self.gan.partial_gp_loss = partial_gp_loss

        self.gan.discriminator_model = Model(
            inputs=[real_img, noise_input, text_input],
            outputs=[real_pred, fake_pred, interpolated_pred]
        )
        return self

    def build_generator_model(self):
        self.gan.discriminator.trainable = False
        self.gan.generator.trainable = True

        # Generar imágenes
        z_input = Input(shape=(self.gan.latent_dim))
        text_input = Input(shape=(self.gan.roberta_shape[0],))
        img = self.gan.generator([z_input, text_input])
        # Predecir si es real
        valid = self.gan.discriminator([img, text_input])
        self.gan.generator_model = Model([z_input, text_input], valid)
        return self
