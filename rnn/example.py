import numpy as np
np.random.seed(1)
from RNN_utils import one_hot_encoding
from DNet.layers import BCELoss
from DNet.optimizers import SGD

input_dim = 27
output_dim = 27
hidden_dim = 50

from DNet.model import NNet
from rnn import RNN

# ---------------------
person_names = open('person_names.txt', 'r').read()
person_names= person_names.lower()
characters = list(set(person_names))

character_to_index = {character:index for index,character in enumerate(sorted(characters))}
index_to_character = {index:character for index,character in enumerate(sorted(characters))}

with open("person_names.txt") as f:
    person_names = f.readlines()

person_names = [name.lower().strip() for name in person_names]
np.random.shuffle(person_names)

# Initialize the model
model = NNet()
# Create the model structure
model.add(RNN(input_dim, output_dim, hidden_dim))

loss = BCELoss()
optim = SGD()


# Train the model
costs = []
num_epochs = 100000

for epoch in range(num_epochs + 1):
    # create the X inputs and Y labels
    index = epoch % len(person_names)
    X = [None] + [character_to_index[ch] for ch in person_names[index]] 
    Y = X[1:] + [character_to_index["\n"]]

    # transform the input X and label Y into one hot enconding.
    X = one_hot_encoding(X, input_dim)
    Y = one_hot_encoding(Y, output_dim)

    model.forward(X)
    cost = model.loss(Y, loss)
    model.backward()
    model.optimize(optim)

    if epoch % 100 == 0:
        print ("Cost after iteration %epoch: %f" %(epoch, cost))
        costs.append(cost)