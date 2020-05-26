import tensorflow as tf
from .data import Flickr8K, RoBERTaTokenizedFlickr8K
from .model import WCGAN, GANAugmenter
import numpy as np

class WCGANTrainer:
    """Configure and run training."""
    
    OUTPUT_LOCATION = "../output/"
    LOG_DIR = "../output/logs"
    
    def __init__(
        self, 
        wcgan: WCGAN, 
        image_dataset: Flickr8K,
        text_dataset: RoBERTaTokenizedFlickr8K,
        output_location: str = OUTPUT_LOCATION,
        log_dir: str = LOG_DIR,
    ):
        self.wcgan = wcgan
        self.images = image_dataset
        self.texts = text_dataset
        self.log_dir = log_dir
        self.output_location = output_location
    
    def train(self, epochs: int = 50):
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.output_location+"_{epoch}",
                save_freq="epoch"
            ),
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
        ]
        self.wcgan.compile(
            tf.keras.optimizers.Adam(learning_rate=0.0003),
            tf.keras.optimizers.Adam(learning_rate=0.0003),
            self.wcgan.wasserstein_loss
        )
        self.wcgan.fit(
            x=[self.images, self.texts], 
            epochs=epochs, 
            callbacks=callbacks
        )
        

class UglyTrainer:
    """Configure and run training horribly coupled."""
    
    OUTPUT_LOCATION = "../output/"
    LOG_DIR = "../output/logs"
    BATCH_SIZE = 36
    
    def __init__(
        self, 
        wcgan: WCGAN, 
        image_dataset: Flickr8K,
        text_dataset: RoBERTaTokenizedFlickr8K,
        output_location: str = OUTPUT_LOCATION,
        log_dir: str = LOG_DIR,
    ):
        self.images = image_dataset
        self.texts = text_dataset
        self.log_dir = log_dir
        self.output_location = output_location
        self.augmenter = GANAugmenter(wcgan)
        self.gan = self._augment_gan()

    def _augment_gan(self):
        return (
            self.augmenter
            .build_discriminator_model()
            .build_generator_model()
            .gan
        )
    
    def compile(self, optimizer):
        self.gan.discriminator_model.compile(
            loss = [self.gan.wasserstein_loss, self.gan.wasserstein_loss, self.gan.partial_gp_loss],
            loss_weights = [1, 1, 10],
            optimizer = optimizer
        )
        self.gan.generator_model.compile(loss=self.gan.wasserstein_loss, optimizer=optimizer)
        
    def save_state(self, ephoch: int):
        self.gan.generator.save(f"../output/generator-{epoch}", save_format="tf")
        self.gan.discriminator.save(f"../output/discriminator-{epoch}", save_format="tf")
        self.gan.generator_model.save(f"../output/generator_model-{epoch}", save_format="tf")
        self.gan.discriminator_model.save(f"../output/discriminator_model-{epoch}", save_format="tf")
        if self.gan.roberta.trainable:
            self.gan.roberta.save(f"../output/roberta-{epoch}", save_format="tf")
    
    def train(self, epochs: int = 50, discriminator_loops: int = 5, buffer_size: int = 500):
        joint_dataset = tf.data.Dataset.zip((self.images, self.texts)).shuffle(buffer_size)
        joint_dataset = iter(joint_dataset)
    
        # Ground truths
        real = -np.ones((self.BATCH_SIZE, 1))
        fake = -real
        dummy = np.zeros((self.BATCH_SIZE, 1))
        
        for epoch in range(epochs):
            for _ in range(discriminator_loops):
                images, text = next(joint_dataset)
                noise = np.randn((batch_size, self.gan.latent_dim))
                d_loss = self.gan.discriminator_model.train_on_batch([images, noise, text])
                
            g_loss = self.gan.generator_model.train_on_batch([noise, text])
            
            self.save_state()
            