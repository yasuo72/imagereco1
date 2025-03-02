import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

# Load the EfficientNet model pre-trained on ImageNet
model = EfficientNetB0(weights='imagenet')


def recognize_image(filepath):
    img = image.load_img(filepath, target_size=(224, 224))  # Resize image
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = preprocess_input(img_array)  # Preprocess image
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Decode top 3 predictions

    return [(pred[1], pred[2]) for pred in decoded_predictions]  # Return label and confidence
