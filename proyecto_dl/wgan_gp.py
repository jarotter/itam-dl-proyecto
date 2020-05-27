import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple
from tensorflow.keras import Model
from transformers import TFRobertaModel
import gcsfs
from.data import Flickr8KImages, RoBERTaTokenizedFlickr8K

class WCGAN(Model):
    """Wasserstein GAN condicional con RoBERTa."""

    def __init__(
        self,
        latent_dim: int,
        img_shape: Tuple[int, int],
        img_channels: int,
        discriminator_extra_steps: int,
        gp_weight: float,
    ):
        super(WCGAN, self).__init__()
        self.latent_dim = latent_dim
        self.img_rows, self.img_cols = img_shape
        self.img_channels = img_channels
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images, text):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([interpolated, text], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        image, text = data

        # Get the batch size
        batch_size = tf.shape(image)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper.
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add gradient penalty to the discriminator loss
        # 6. Return generator and discriminator losses as a loss dictionary.

        # Train discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(
                    [random_latent_vectors, text], 
                    training=True
                )
                # Get the logits for the fake images
                fake_logits = self.discriminator(
                    [fake_images, text], 
                    training=True
                )
                # Get the logits for real images
                real_logits = self.discriminator(
                    [image, text], 
                    training=True
                )

                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, image, fake_images, text)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator now.
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(
                [random_latent_vectors, text],
                training=True
            )
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(
                [generated_images, text],
                training=True
            )
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

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
        discriminator_extra_steps: int = 5,
        gp_weight=10.0
    ):
        self._model = WCGAN(latent_dim, img_shape, img_channels, discriminator_extra_steps, gp_weight)
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

        img_input = layers.Input(shape=self._model.img_shape, name="discriminator_img")
        img_params = layers.Conv2D(
            filters=16,
            kernel_size=kernel_size,
            strides=2,
            input_shape=self._model.img_shape,
            padding=padding,
        )(img_input)
        img_params = layers.LeakyReLU(alpha=leaky_alpha)(img_params)
        img_params = layers.Dropout(dropout_probability)(img_params)
        img_params = layers.Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=2,
            input_shape=self._model.img_shape,
            padding=padding,
        )(img_params)
        img_params = layers.LeakyReLU(alpha=leaky_alpha)(img_params)
        img_params = layers.Dropout(dropout_probability)(img_params)
        img_params = layers.Conv2D(
            filters=64, kernel_size=kernel_size, strides=2, padding=padding
        )(img_params)
        img_params = layers.LeakyReLU(alpha=leaky_alpha)(img_params)
        img_params = layers.Dropout(dropout_probability)(img_params)
        img_params = layers.Conv2D(
            filters=128, kernel_size=kernel_size, strides=2, padding=padding
        )(img_params)
        img_params = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(img_params)
        img_params = layers.LeakyReLU(alpha=leaky_alpha)(img_params)
        img_params = layers.Dropout(dropout_probability)(img_params)
        img_params = layers.Reshape((128 * 5 * 5,))(img_params)

        # Aquí voy a sustituir lo que hacen de las labels
        # para utilizar bert
        sentence_length = self._model.roberta_shape[0]
        text_params = layers.Input(shape=(sentence_length,), dtype=tf.int64, name="discriminator_text")
        roberta_embeddings = self._model.roberta(text_params)[0]
        roberta_embeddings = layers.Dense(128)(roberta_embeddings)
        roberta_embeddings = layers.Reshape((128 * 5 * 5,))(roberta_embeddings)

        joint_params = layers.concatenate((roberta_embeddings, img_params))
        joint_params = layers.Dense(512)(joint_params)
        joint_params = layers.LeakyReLU(alpha=leaky_alpha)(joint_params)
        joint_params = layers.Dropout(dropout_probability)(joint_params)
        joint_params = layers.Dense(128)(joint_params)
        joint_params = layers.LeakyReLU(alpha=leaky_alpha)(joint_params)
        joint_params = layers.Dense(32)(joint_params)
        joint_params = layers.Dropout(dropout_probability)(joint_params)
        joint_params = layers.LeakyReLU(alpha=leaky_alpha)(joint_params)
        output = layers.Dense(1)(joint_params)

        model = Model([img_input, text_params], output, name="discriminator")
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
        noise_input = layers.Input(shape=(self._model.latent_dim,), name="generator_noise")
        sentence_length = self._model.roberta_shape[0]
        text_params = layers.Input(shape=(sentence_length,), dtype=tf.int64, name="generator_text")
        roberta_embeddings = self._model.roberta(text_params)[0]
        roberta_embeddings = layers.Bidirectional(layers.LSTM(64))(roberta_embeddings)

        adjusted_input = layers.multiply((roberta_embeddings, noise_input))

        # Construir imagen
        params = layers.Dense(128 * 8 * 8)(adjusted_input)
        params = layers.LeakyReLU(alpha=leaky_alpha)(params)
        params = layers.Reshape((8, 8, 128))(params)
        params = layers.UpSampling2D()(params)
        params = layers.Conv2D(128, kernel_size=kernel_size, padding=padding)(params)
        params = layers.BatchNormalization(momentum=momentum)(params)
        params = layers.LeakyReLU(alpha=leaky_alpha)(params)
        params = layers.UpSampling2D()(params)
        params = layers.Conv2D(64, kernel_size=kernel_size, padding=padding)(params)
        params = layers.BatchNormalization(momentum=momentum)(params)
        params = layers.LeakyReLU(alpha=leaky_alpha)(params)
        params = layers.UpSampling2D()(params)
        params = layers.Conv2D(
            self._model.img_channels, kernel_size=kernel_size, padding="same"
        )(params)
        img = layers.Activation("tanh")(params)

        model = Model([noise_input, text_params], img, name="generator")
        self._model.generator = model
        return self

    def return_(self):
      return self._model


class WCGANTrainer:

    LOG_DIR = "gs://tti-roberta-wcgan/gan-tensorboard"
    MODEL_DIR = "gs://tti-roberta-wcgan/gan-checkpoints/second_train/"

    @staticmethod
    def discriminator_loss(real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    @staticmethod
    def generator_loss(fake_img):
        return -tf.reduce_mean(fake_img)


    def __init__(
        self, 
        gan: WCGAN,
        log_dir: str = LOG_DIR, 
        model_dir = MODEL_DIR,
    ):
        self.images = Flickr8KImages()
        self.text = RoBERTaTokenizedFlickr8K()
        self.histories = []
        self.log_dir = log_dir
        self.out_dir = model_dir
        self.gan = gan

    def _build_joint_data(self, batch_size: int):
        return (
            tf.data.Dataset
            .zip((self.images.test, self.text.test))
            .batch(batch_size)
        )

    def train(self, epochs: int = 50, batch_size: int=36):
        data = self._build_joint_data(batch_size)
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.out_dir+"checkpoint_{epoch}",
                save_freq="epoch"
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                update_freq="epoch"
            )
        ]
        history = self.gan.fit(data, epochs=epochs, callbacks=callbacks)
        self.histories.append(history)
        return history

