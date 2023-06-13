import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Step 1: Instructions and Help
print("Welcome to the Expert System!")
print("This system allows you to explore the MedMNIST Breast MNIST dataset and train a neural network model.")

# Step 2: Background Information
print("The MedMNIST Breast MNIST dataset contains 28x28 grayscale images of breast pathology.")
print("The dataset consists of 2 classes, representing benign and malignant breast tumors.")

# Step 3: Load Dataset
url = "https://zenodo.org/record/6496656/files/breastmnist.npz"
response = requests.get(url)
data = np.load(BytesIO(response.content))
X_train, y_train, X_test, y_test = data['train_images'], data['train_labels'], data['test_images'], data['test_labels']

# Step 4: Preview Image Data
sample_image, sample_label = X_train[0], y_train[0]
plt.imshow(sample_image, cmap='gray')
plt.title(f"Label: {sample_label}")
plt.axis('off')
plt.show()

# Step 5: Montage of Images
num_images = 10
montage_images = X_train[:num_images]
montage_labels = y_train[:num_images]
montage = np.hstack(montage_images)
plt.imshow(montage, cmap='gray')
plt.title("Montage of Images")
plt.axis('off')
plt.show()

# Step 6: Generate Histogram of Dataset
labels = y_train
plt.hist(labels, bins=2)
plt.title("Histogram of Dataset")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.xticks(range(2))
plt.show()

# Step 7: View Separate RGB Color Channels (for grayscale images)
rgb_image = np.stack([sample_image] * 3, axis=-1)
r_channel, g_channel, b_channel = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
fig, axes = plt.subplots(1, 3)
axes[0].imshow(r_channel, cmap='gray')
axes[0].set_title("Red Channel")
axes[0].axis('off')
axes[1].imshow(g_channel, cmap='gray')
axes[1].set_title("Green Channel")
axes[1].axis('off')
axes[2].imshow(b_channel, cmap='gray')
axes[2].set_title("Blue Channel")
axes[2].axis('off')
plt.show()

# Step 8: Train Neural Network
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)
train_labels = torch.from_numpy(y_train.squeeze()).long()
test_labels = torch.from_numpy(y_test.squeeze()).long()
trainset = TensorDataset(torch.from_numpy(X_train).float(), train_labels)
testset = TensorDataset(torch.from_numpy(X_test).float(), test_labels)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Step 9: Train and Evaluate the Model
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
test_losses = []
train_accuracy = []
test_accuracy = []

def train(model, trainloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)  
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(trainloader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracy.append(train_acc)

def evaluate(model, testloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(testloader)
    test_acc = correct / total
    test_losses.append(test_loss)
    test_accuracy.append(test_acc)

    return test_acc

# Step 10: Train the model
num_epochs = 10
for epoch in range(num_epochs):
    train(model, trainloader, optimizer, criterion)
    acc = evaluate(model, testloader, criterion)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Accuracy: {acc:.4f}")

# Step 11: Plot Training and Testing Accuracy
plt.plot(train_accuracy, label="Training Accuracy")
plt.plot(test_accuracy, label="Testing Accuracy")
plt.title("Training and Testing Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Step 12: Expert System Style Q/A for Patient Follow-up
def expert_system(input_image):
    model.eval()
    input_image = input_image.reshape(1, -1)
    with torch.no_grad():
        output = model(torch.from_numpy(input_image).float())
        _, predicted = torch.max(output, 1)
    
    if predicted.item() == 0:
        return "The tumor is classified as benign. No immediate intervention is required."
    else:
        return "The tumor is classified as malignant. Please consult a healthcare professional for further evaluation and treatment."

# Provide example input image for expert system
example_input = X_test[1]
print(expert_system(example_input))
