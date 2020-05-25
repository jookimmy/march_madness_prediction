import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from makedata import make_data 
import matplotlib.pyplot as plt

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        super(NeuralNet, self).__init__()
        self.lc1 = nn.Linear(in_size, 30)
        self.lc2 = nn.Linear(30, out_size)
        self.activation = nn.Sigmoid()
        self.loss_fn = loss_fn
        self.net = nn.Sequential(
            self.lc1,
            self.activation,
            self.lc2,
            self.activation)
        self.optimizer = optim.SGD(self.net.parameters(), lr=lrate)

    def forward(self, x):
        return self.net(x)

    def step(self, x, y):
        self.optimizer.zero_grad()
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()

        return loss, output

    def fit(self, train_set, train_labels, matches, win_loss, n_iter=100, batch_size=4):
        losses = []

        for left in range(0, 67, batch_size):
            right = (left+batch_size)%67 if (left+batch_size) < 67 else 67
            print(left, right)
            inputs, labels, match = train_set[left:right], train_labels[left:right], matches[left:right]

            loss, output = self.step(inputs, labels)
            losses.append(loss)

            actual = []
            predicted = []

            for i in range(len(inputs)):
                print(f'Match Played: {match[i]} Predicted Outcome: {win_loss[output.max(0)[1][i]]} Actual Outcome: {win_loss[labels[i]]}')
                predicted.append(win_loss[output.max(0)[1][i]])
                actual.append(win_loss[labels[i]])

            print(f'Loss: {losses[-1]}')
            time.sleep(.1)
        return losses, predicted, actual


def plot_loss(losses):
    """Plots the losses of the network over the training phase
    """
    plt.figure()
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("binary cross-entropy loss")
    plt.plot(losses)
    plt.show()

if __name__ == "__main__":
    stats = ['ADJOE','ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D', 'TOR', 'TORD', 'ORB','DRB', 'FTR', 'FTRD', '2P_O','2P_D','3P_O','3P_D', 'ADJ_T','WAB','SEED']
    win_loss = ['Blowout win', 'Close win', 'Blowout loss', 'Close loss']
    train_set, train_labels, matches = make_data('15', stats)
    train_set = np.array(train_set)

    train_set = torch.tensor(train_set,dtype=torch.float32)
    train_labels = torch.tensor(train_labels,dtype=torch.int64)

    train_set = (train_set - train_set.mean())/train_set.std()
    loss_fn = nn.CrossEntropyLoss()
    model = NeuralNet(1, loss_fn, 18, 4)

    losses = []
    predicted = []
    actual = []

    for epoch in range(50):
        loss, pred, act = model.fit(train_set, train_labels, matches, win_loss)
        losses += loss
        predicted += pred
        actual += act
    
    accuracy = sum(1 for x,y in zip(actual,predicted) if x == y) / len(actual)
    print(f'Accuracy was: {accuracy}')

    plot_loss(losses)

    torch.save(model.state_dict(), f'{accuracy}model')


