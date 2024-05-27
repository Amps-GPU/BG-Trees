from dataclasses import dataclass
import tensorflow as tf


@dataclass
class Settings:
    use_gpu: bool = False
    D: int = 4

    def run_tf_eagerly():
        tf.config.run_functions_eagerly(True)


settings = Settings()

# settings.run_tf_eagerly()
