import tensorflow as tf
from .data import Flickr8K, RoBERTaTokenizedFlickr8K
from .model import WCGAN

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
        