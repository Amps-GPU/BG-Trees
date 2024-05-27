from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass
class Settings:
    use_gpu: bool = False
    D: int = 4
    dtype: type = np.int64

    def run_tf_eagerly(self):
        tf.config.run_functions_eagerly(True)


settings = Settings()

# settings.run_tf_eagerly()
