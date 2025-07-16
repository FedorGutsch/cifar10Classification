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

    def _prepare_dataset(self, images, labels, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        
       
        dataset = dataset.map(
            lambda img, lbl: (tf.cast(img, tf.float32) / 255.0, tf.one_hot(tf.squeeze(lbl), config.NUM_CLASSES)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(images))
        
        dataset = dataset.batch(config.BATCH_SIZE)
        
       
        if shuffle:
            dataset = dataset.map(
                lambda x, y: (self.data_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def get_train_dataset(self):
        """Возвращает тренировочный датасет"""  
        return self._prepare_dataset(self.train_images, self.train_labels, shuffle=True)

    def get_test_dataset(self):
        """Возвразает тестовый датасет"""
        return self._prepare_dataset(self.test_images, self.test_labels)