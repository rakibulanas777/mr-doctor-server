import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

from nltk.stem.porter import PorterStemmer

with open('intents.json', 'r') as f:
    intents = json.load(f)

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Prepare the dataset
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stem and remove duplicates
all_words = [stem(w) for w in all_words if w not in ['?', '!', '.', ',']]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 300

# Load model, loss, optimizer
model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i in range(len(X_train)):
        X = torch.tensor(X_train[i], dtype=torch.float32)
        y = torch.tensor(y_train[i], dtype=torch.long)

        # Forward pass
        output = model(X)
        loss = criterion(output.unsqueeze(0), y.unsqueeze(0))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def predict_class(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X, dtype=torch.float32)
    output = model(X)
    _, predicted = torch.max(output, dim=0)
    tag = tags[predicted.item()]
    return tag

def get_response(intent, intents):
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break
    intent = predict_class(sentence)
    response = get_response(intent, intents)
    print(f"Bot: {response}")


# Save model
torch.save(model.state_dict(), "chatbot_model.pth")


