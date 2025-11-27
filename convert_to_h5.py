import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Lambda

# -------------------------------
# CONFIG
# -------------------------------
saved_model_dir = os.path.abspath(r"C:\Users\Dominique\Downloads\Pawprint\backend\dog_model")
output_keras_file = "dog_model_clean.keras"  # <--- save as .keras

# -------------------------------
# Load SavedModel
# -------------------------------
loaded_model = tf.saved_model.load(saved_model_dir)
signature_name = "serving_default"
infer = loaded_model.signatures[signature_name]
print("Output keys:", list(infer.structured_outputs.keys()))

# Pick first output key
output_key = list(infer.structured_outputs.keys())[0]
output_shape = infer.structured_outputs[output_key].shape[1:]  # remove batch dim

# -------------------------------
# Wrap as Keras model
# -------------------------------
inputs = Input(shape=(224, 224, 3))

# Use Lambda with output_shape specified
outputs = Lambda(
    lambda x: infer(tf.cast(x, tf.float32))[output_key],
    output_shape=output_shape
)(inputs)

keras_model = Model(inputs, outputs)

# -------------------------------
# Save as native Keras format
# -------------------------------
keras_model.save(output_keras_file)
print(f"✅ Model saved in native Keras format: {output_keras_file}")
