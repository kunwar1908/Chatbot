import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random
from keras import Sequential
from keras import layers
from keras import Input
from keras import optimizers

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file
data_file = open('C:/Users/kunwa/Python/Projects/chatbot-python-project-data-codes/intents.json').read()
intents = json.loads(data_file)

# Initialize lists
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
        # Add documents in the corpus
        documents.append((w, intent['tag']))
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Sort classes
classes = sorted(list(set(classes)))

# Print document, class, and word statistics
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
# Create an empty array for our output
output_empty = [0] * len(classes)

# Create the bag of words for each pattern
for doc in documents:
    # Initialize our bag of words
    bag = []
    # List of tokenized words for the pattern
    pattern_words = doc[0]
    # Lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # Output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# Shuffle and convert to numpy arrays
random.shuffle(training)
training = np.array(training, dtype=object)

# Create train and test lists. X - patterns, Y - intents
train_x = np.array([t[0] for t in training])
train_y = np.array([t[1] for t in training])

# Print shapes
print("Shape of train_x:", train_x.shape)
print("Shape of train_y:", train_y.shape)

# Create model
model = Sequential()
model.add(Input(shape=(len(words),)))  # Input layer specifying the shape
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(classes), activation='softmax'))

# Compile model
sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Fit the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose='1')
model.save('chatbot_model.h5')

print("Model created")

# Print model summary
model.summary()
