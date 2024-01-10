import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, start_index, end_index):
        self.data = pd.read_csv(csv_file, skiprows=1)
        self.data = self.data[start_index: end_index]
        self.data.head()
        print(self.data)



        # Assuming 'Date' is in 'YYYY-MM-DD' format
        # Reverse the order of the 'Date' Column - NEED TO DO
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        # Identify the reference event's timestamp and convert to days
        reference_event_timestamp = pd.to_datetime('2022-01-07 00:00:00')
        # Calculate the time since the reference event in integers
        self.data['Date'] = (self.data['Date'] - reference_event_timestamp).dt.days
        self.data['Numeric_Date'] = self.data['Date']
        #self.data['Numeric_Date'] = (self.data['Date'] - pd.Timestamp("2022-01-11")) // pd.Timedelta('1d')

        self.features = torch.tensor(self.data['Numeric_Date'].values, dtype=torch.float32).view(-1, 1, 1)
        self.labels = torch.tensor(self.data['Number of  reported results'].values, dtype=torch.float32)
        print(self.features)
        print(self.labels)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)

        # Forward pass through GRU layer
        out, _ = self.gru(x, h0)

        # Take the output from the last time step
        out = out[:, -1, :]

        # Forward pass through fully connected layer
        out = self.fc(out)

        # Remove unnecessary dimensions
        out = out.squeeze(1)

        return out


# Create a dataset instance
start_index = 0
end_index = 300

dataset = CustomDataset('problemcData.csv', start_index, end_index)

# Example usage
input_size = 1
hidden_size = 8
num_layers = 3
output_size = 1
batch_size = 10
sequence_length = 7

# Create a DataLoader for batching
dataloader = DataLoader(dataset, batch_size, shuffle=True)

# Create GRU model
model = GRUModel(input_size, hidden_size, num_layers, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 15  # Adjust the number of epochs as needed
for epoch in range(num_epochs):
    for batch in dataloader:
        input_data, labels = batch
        output = model(input_data)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model weights
torch.save(model.state_dict(), 'path_to_your_trained_model_weights.pth')

# Example usage with DataLoader after training
with torch.no_grad():
    for batch in dataloader:
        input_data, labels = batch
        output = model(input_data)

print("Output Shape:", output.shape)
print(output)
output_values = output.numpy()
print(output_values)

# Plot the predicted values
plt.plot(output.detach().numpy(), label='Predicted')
plt.plot(labels.numpy(), label='True Labels')
plt.legend()
plt.show()
