import pathlib
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

batch_size = 32
img_height = 180
img_width = 180
seed=123

test_dir = pathlib.Path("test")
print("Test images: {}".format(len(list(test_dir.glob('*')))))

for img_path in list(test_dir.glob('*')):
    print(img_path)
    img = keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    print(img_array)

