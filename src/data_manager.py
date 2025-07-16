# src/data_manager.py

import tensorflow as tf
from src import config

class Cifar10DataManager:
 

    def __init__(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = tf.keras.datasets.cifar10.load_data()
        
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                              'dog', 'frog', 'horse', 'ship', 'truck']
      