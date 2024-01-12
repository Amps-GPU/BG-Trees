from os import chdir, curdir
from pathlib import Path

import tensorflow as tf

# Loading of the modules
# For the time being, go to the folder where I'm compiling and take them from there
_modules_folder = Path(__file__).parent
_orig_folder = Path(curdir).absolute()

chdir(_modules_folder)
dot_product_module = tf.load_op_library("./dot_product.so")
chdir(_orig_folder)


# Functions
@tf.function
def wrapper_dot_product(x, y):
    ret = dot_product_module.dot_product(x, y)
    return ret
