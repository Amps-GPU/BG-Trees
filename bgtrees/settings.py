from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass
class Settings:
    use_gpu: bool = False
    D: int = 4
    dtype: type = np.int64
    p: int = 2**31 - 19
    alternating_metric: bool = False  # debug setting e.g. to match Caravel, leave to false

    # Tensorflow settings
    def run_tf_eagerly(self):
        tf.config.run_functions_eagerly(True)

    def executing_eagerly(self):
        return tf.executing_eagerly()

    @property
    def tf_p(self):
        return tf.cast(self.p, dtype=settings.dtype)


settings = Settings()

# settings.run_tf_eagerly()
