import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
import os

# Allow import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.model import initialize_model
from shared.utils import get_device
from config.loader import config_loader


class EdgeTrainer:
    def __init__(self, device_id):
        self.device_id = device_id
        self.device = get_device()

        self.learning_rate = config_loader.get("training", "learning_rate")
        self.batch_size = config_loader.get("training", "batch_size")
        self.local_epochs = config_loader.get("training", "local_epochs")

        self.model = initialize_model(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_data(self, X_train, y_train, X_test, y_test):
        self.train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True
        )

        self.test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=self.batch_size,
            shuffle=False
        )

    def train(self):
        self.model.train()

        for epoch in range(self.local_epochs):
            total_loss = 0

            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"[{self.device_id}] Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    def evaluate(self):
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)

                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = 100 * correct / total
        print(f"[{self.device_id}] Accuracy: {accuracy:.2f}%")

        return accuracy

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)
