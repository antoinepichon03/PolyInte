import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import argparse

from torchvision import transforms
from typing import List, Tuple, Union
from tqdm.auto import tqdm
from q1 import (
    Layer,
    Dense,
    SoftmaxLayer,
    TanhLayer,
    ReLULayer,
    CrossEntropyLossLayer
)
from q2 import MLP

"""INSTRUCTIONS TO RUN THIS SCRIPT
This script can be run from the terminal by running the following command:
    python fmnist.py **ARGS
where **ARGS are the arguments to the script. The arguments are as follows:
    --model: The model to use for training. Choose from 'pytorch' or 'custom'
        (default usage: both models are trained)
    --batch_size: Batch size for training 
    --lr: Learning rate for training 
    --epochs: Number of epochs to train the model 
    --val-perc: Percentage of the training data to use for validation (default: 0.2)
    --plot: Whether to plot the training loss and validation accuracy (default: False)

Example:
    python fmnist.py --model custom --batch_size 128 --lr 0.001 --epochs 20 --val-perc 0.2 --plot
"""

## SET GLOBAL SEED
## Do not modify this for reproducibility
np.random.seed(33)

class FashionMNIST:
    """Data loading for Fashion MNIST dataset"""

    def __init__(self, batch_size, val_perc: float = 0.2):
        self.data_seed = 6666
        np.random.seed(self.data_seed)
        torch.manual_seed(self.data_seed)
        torch.cuda.manual_seed(self.data_seed)
        self.generator = torch.Generator().manual_seed(self.data_seed)

        self.batch_size = batch_size
        self.val_perc = val_perc
        self.load_data()

    def load_data(self):
        """
        Load the FashionMNIST dataset.
        """
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
        ])

        self.train_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform
        )
        if self.val_perc is not None and self.val_perc > 0:
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset,
                [
                    int((1 - self.val_perc) * len(self.train_dataset)),
                    int(self.val_perc * len(self.train_dataset))
                ],
                generator=self.generator
            )
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=4, pin_memory=True,
                generator=self.generator
            )
            
        self.test_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
            generator=self.generator
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
            generator=self.generator
        )
        

class PytorchMLPFashionMNIST(nn.Module):
    def __init__(
        self,
        input_size=28*28,
        hidden_size1=256,
        hidden_size2=128,
        output_size=10
    ):
        """
        A simple MLP model for Fashion MNIST classification.
        """
        super(PytorchMLPFashionMNIST, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
            # NOTE: !!!!!! NOTE !!!!!!
            # pytorch's CrossEntropyLoss includes Softmax operation so we do not do it here.
            # nn.Softmax(dim=1) 
        )

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 28 * 28)
        # Forward pass
        return self.layers(x)

    def train(
        self,
        data: FashionMNIST,
        criterion: nn.Module=nn.CrossEntropyLoss(),
        lr: float=0.001,
    ):
        """
        Trains one epoch of the model.
        Validates the model on the validation set if available.
        Args:
            data: FashionMNIST object
            criterion: Loss function
            lr: Learning rate
        Returns:
            train_loss: Average training loss
            val_accuracy: Validation accuracy       
        """
        self.optimizer = self.optimizer \
            if hasattr(self, 'optimizer') else \
               optim.SGD(self.parameters(), lr=lr)

        running_loss = 0.0
        correct = 0
        total = 0

        # train
        for images, labels in data.train_loader:
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # Forward pass
            outputs = self(images)
            # Calculate loss
            loss = criterion(outputs, labels)
            # Calculate loss grad
            loss.backward()
            # Update weights
            self.optimizer.step()
            outputs = outputs.detach().numpy()
            running_loss += loss.item()
        train_loss = running_loss / len(data.train_loader)

        # Validate
        if data.val_perc is not None and data.val_perc > 0:
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in data.val_loader:
                    labels = labels.numpy()
                    outputs = self(images)
                    outputs = outputs.detach().numpy()
                    predicted = np.argmax(outputs, axis=1)
                    total += labels.shape[0]
                    correct += (predicted == labels).sum().item()
            val_accuracy = correct / total

        return train_loss, val_accuracy

    def test(self, data: FashionMNIST):
        """
        Tests the model on the test set.
        Args:
            data: FashionMNIST object
        Returns:
            test_accuracy: Test accuracy
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data.test_loader:
                labels = labels.numpy()
                outputs = self(images)
                outputs = outputs.detach().numpy()
                predicted = np.argmax(outputs, axis=1)
                total += labels.shape[0]
                correct += (predicted == labels).sum().item()

        return correct/total

class CustomMLPFashionMNIST(Layer):
    def __init__(
        self,
        input_size=28*28,
        hidden_size1=256,
        hidden_size2=128,
        output_size=10
    ):
        super(CustomMLPFashionMNIST, self).__init__()
        self.layers = MLP([
            Dense(input_size, hidden_size1),
            ReLULayer(),
            Dense(hidden_size1, hidden_size2),
            ReLULayer(),
            Dense(hidden_size2, output_size),
            SoftmaxLayer()
        ])

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.layers(x)

    def backward(self, grad):
        return self.layers.backward(grad)

    def update(self, lr):
        self.layers.update(lr)

    def train(
        self,
        data: FashionMNIST,
        criterion: Layer=CrossEntropyLossLayer(),
        lr: float=0.001,
    ):
        """
        Trains one epoch of the model.
        Validates the model on the validation set if available.
        Args:
            data: FashionMNIST object
            criterion: Loss function
            lr: Learning rate
        Returns:
            train_loss: Average training loss
            val_accuracy: Validation accuracy
        """
        running_loss = 0.0
        correct = 0
        total = 0

        # train
        for images, labels in data.train_loader:
            labels = labels.numpy()
            # Forward pass
            outputs = self(images.view(images.size(0), -1).numpy())
            # Calculate loss
            loss = criterion(outputs, labels)
            # Calculate loss grad
            loss_grad = criterion.backward(1)
            # Backward pass
            self.backward(loss_grad)
            # Update weights
            self.update(lr)
            running_loss += loss.item()

        train_loss = running_loss / len(data.train_loader)

        # Validate
        if data.val_perc is not None and data.val_perc > 0:
            correct = 0
            total = 0
            for images, labels in data.val_loader:
                labels = labels.numpy()
                outputs = self(images.view(images.size(0), -1))
                predicted = np.argmax(outputs, axis=1)
                total += labels.shape[0]
                correct += (predicted == labels).sum().item()
            val_accuracy = correct / total

        return train_loss, val_accuracy

    def test(self, data: FashionMNIST):
        """
        Tests the model on the test set.
        Args:
            data: FashionMNIST object
        Returns:
            test_accuracy: Test accuracy
        """
        correct = 0
        total = 0
        for images, labels in data.test_loader:
            labels = labels.numpy()
            outputs = self(images.view(images.size(0), -1).numpy())
            predicted = np.argmax(outputs, axis=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

        return correct/total
            


def plot(
    train_losses,
    val_accuracies,
    title='Training Loss and Validation Accuracy'
):
    """
    Plots the training loss and validation accuracy.
    Args:
        train_losses: List of training losses
        val_accuracies: List of validation accuracies
        title: Title of the plot
    """
    ## Plotting the train loss and accuracy
    epochs = len(train_losses)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(
        np.arange(1, epochs+1),
        train_losses,
        color='blue'
    )
    axes[0].set_title('Training loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid()
    
    axes[1].plot(
        np.arange(1, epochs+1),
        val_accuracies,
        color='red'
    )
    axes[1].set_title('Val accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid()
    
    axes[0].set_xticks(range(0, epochs+1, 2))
    axes[1].set_xticks(range(0, epochs+1, 2))
    fig.tight_layout()

    plt.suptitle(title)
    
    plt.show()


if __name__ == '__main__':
    # pip install matplotlib tqdm torchvision torch
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--models',
        nargs='+',
        default=['pytorch', 'custom'],
        help='Models to train. Options: pytorch, custom'
    )
    parser.add_argument('--lr', type=float, default=1, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument(
        '--val-perc', type=float, default=0.2,
        help='Validation percentage'
    )
    parser.add_argument(
        '--plot', action='store_true',
        default=False,
        help='Plot training stats'
    )

    args = parser.parse_args()
    
    # Hyperparameters
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    val_perc = args.val_perc

    # initialize the dataset
    dataset = FashionMNIST(batch_size=batch_size, val_perc=val_perc)

    # train models
    for mod in args.models:
        if mod == 'pytorch':
            model = PytorchMLPFashionMNIST()
        else:
            model = CustomMLPFashionMNIST()
        
        # train model
        pbar = tqdm(
            range(epochs),
            desc=f'Training {mod}',
            position=0,
            leave=True
        )
        train_losses = []
        val_accuracies = []

        ## Training loop
        for epoch in pbar:
            train_loss, val_acc = model.train(dataset, lr=lr)
            pbar.set_postfix(
                train_loss=f'{train_loss:.4f}',
                val_acc=f'{val_acc:.2%}'
            )
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)
            
        test_acc = model.test(dataset)

        ## Print the test accuracy
        print(f'[{mod.upper()}] Test accuracy: {test_acc:.2%}')

        ## Plot the training loss and accuracy
        if args.plot:
            plot(
                train_losses, val_accuracies,
                title=f'[{mod.upper()}] Training Loss and Validation Accuracy'
            )
