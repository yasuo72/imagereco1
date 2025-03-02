from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import model_from_json

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Convert the model to JSON
model_json = model.to_json()

# Save the model as a JSON file
with open("intents.json", "w") as json_file:
    json_file.write(model_json)