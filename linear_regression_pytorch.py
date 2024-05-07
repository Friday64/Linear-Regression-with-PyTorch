import torch
import torch.nn as nn
import torch.optim as optim

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Create a dataset (y = 2 * x + 1)
inputs = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32).to(device)
outputs = torch.tensor([[3], [5], [7], [9]], dtype=torch.float32).to(device)

# Step 3: Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Initialize the model and move it to the GPU
model = LinearRegressionModel(1, 1).to(device)

# Step 4: Set up the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 5: Train the model
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = criterion(predictions, outputs)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Step 6: Make predictions using the trained model
test_input = torch.tensor([[5]], dtype=torch.float32).to(device)
test_output = model(test_input)
print(f"Predicted value for input 5: {test_output.item():.4f}")
