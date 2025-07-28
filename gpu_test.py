import tensorflow as tf

print(tf.__version__)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.test.is_built_with_cuda():
    print("TensorFlow is built with CUDA support.")
else:
    print("TensorFlow is NOT built with CUDA support.")