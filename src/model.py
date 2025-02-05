import torch
import torch.nn as nn
import torch.optim as optim

class IncomePredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 8, learning_rate: float = 0.01):
        super(IncomePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 2 input features -> 8 hidden units
        self.fc2 = nn.Linear(hidden_size, 1)  # 8 hidden units -> 1 output
        self.relu = nn.ReLU()

        # Loss and Optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def sgd(self, train_data, targets, epochs=100):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self(train_data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            if epoch % (epochs // 10) == 0:  # Print progress every 10% of epochs
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')


    def evaluate(self, test_input):
        with torch.no_grad():
            prediction = self(test_input).item()
        return prediction