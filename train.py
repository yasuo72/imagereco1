import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents
        documents.append((w, intent['tag']))
        # Add classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create bag of words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # Create output row
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Debugging: Check the shape and type of training data
print(f"Sample training data: {training[0]}")
print(f"Length of training list: {len(training)}")
print(f"Shape of bags and output rows: {len(training[0][0])}, {len(training[0][1])}")

# Check the shapes and consistency of training data
for i, (bag, output) in enumerate(training):
    if len(bag) != len(words):
        print(f"Mismatch in bag length at index {i}: {len(bag)} (expected {len(words)})")
    if len(output) != len(classes):
        print(f"Mismatch in output length at index {i}: {len(output)} (expected {len(classes)})")

# Shuffle and convert to numpy arrays
random.shuffle(training)

try:
    # Convert lists to numpy arrays
    train_x = np.array([item[0] for item in training])
    train_y = np.array([item[1] for item in training])
    print("Training data converted to numpy arrays successfully.")
except Exception as e:
    print(f"Error converting to numpy arrays: {e}")
    raise  # Re-raise the exception to stop execution if conversion fails

# Ensure train_x and train_y are defined before proceeding
if 'train_x' not in locals() or 'train_y' not in locals():
    raise ValueError("train_x and train_y must be defined")

print("Training data created")

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=600, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5')

print("Model created and saved as chatbot_model.h5")
