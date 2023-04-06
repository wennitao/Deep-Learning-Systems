import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    residualPreBlock = nn.Sequential (nn.Linear (dim, hidden_dim), norm(hidden_dim), nn.ReLU(), nn.Dropout (drop_prob), nn.Linear (hidden_dim, dim), norm(dim))
    resBlock = nn.Sequential (nn.Residual (residualPreBlock), nn.ReLU())
    return resBlock
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = [nn.Linear (dim, hidden_dim), nn.ReLU()]
    for _ in range (num_blocks):
      modules.append (ResidualBlock (hidden_dim, hidden_dim//2, norm=norm, drop_prob=drop_prob))
    modules.append (nn.Linear (hidden_dim, num_classes))
    ResNet = nn.Sequential (*modules)
    return ResNet
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
      model.train()
    else:
      model.eval()
    testNum, hitNum, batchNum, lossSum = 0, 0, 0, 0
    for i, batch in enumerate (dataloader):
      batchNum += 1
      batch_x, batch_y = batch[0], batch[1]
      model_y = model (batch_x)
      testNum += batch_x.shape[0]
      hitNum += np.sum (np.argmax (model_y.numpy(), axis=1) == batch_y.numpy())
      loss = nn.SoftmaxLoss ()(model_y, batch_y)
      lossSum += loss.numpy()
      if opt:
        opt.reset_grad()
        loss.backward()
        opt.step()
    return ((testNum - hitNum) / testNum, lossSum / batchNum)
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset (data_dir + "/train-images-idx3-ubyte.gz", data_dir + "/train-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader (dataset = train_dataset, batch_size = batch_size, shuffle = True)
    model = MLPResNet (784, hidden_dim = hidden_dim)
    model.train()
    for epoch_num in range (epochs):
      if optimizer != None:
        training_accuracy, training_loss = epoch (train_dataloader, model, opt = optimizer(model.parameters(), lr = lr, weight_decay = weight_decay))
      else:
        training_accuracy, training_loss = epoch (train_dataloader, model)
      print ('TrainingErr: ', training_accuracy, ' TrainingLoss: ', training_loss)

    eval_dataset = ndl.data.MNISTDataset (data_dir + "/t10k-images-idx3-ubyte.gz", data_dir + "/t10k-labels-idx1-ubyte.gz")
    eval_dataloader = ndl.data.DataLoader (dataset = eval_dataset, batch_size = batch_size) 
    model.eval()
    eval_accuracy, eval_loss = epoch (eval_dataloader, model)
    print (training_accuracy, training_loss, eval_accuracy, eval_loss)
    return (training_accuracy, training_loss, eval_accuracy, eval_loss)
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
