# src/data_manager.py

import tensorflow as tf
from src import config

class Cifar10DataManager:
 

    def __init__(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = tf.keras.datasets.cifar10.load_data()
        
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                              'dog', 'frog', 'horse', 'ship', 'truck']
        
        self._build_augmentation_pipeline()

    
    def _build_augmentation_pipeline(self):
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS)),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])

    