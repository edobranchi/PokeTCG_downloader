import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image as tf_image


# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='pokemon_card_classifier.tflite')
interpreter.allocate_tensors()



def preprocess_image_tflite(image_path):
    # Load the image
    img = tf_image.load_img(image_path, target_size=(224, 224))
    img_array = tf_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array


def predict_tflite(image,index_to_label):
    # Get tensor details (input and output tensors)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the tensor data (input image)
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get the output predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)
    print(f"Predicted class: {predicted_class}")
    # Map back to the class name
    predicted_label = index_to_label[predicted_class[0]]
    print(f"Predicted class: {predicted_label}")






