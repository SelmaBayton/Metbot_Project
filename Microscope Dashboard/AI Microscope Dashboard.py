# Documentation to Help

import matplotlib
matplotlib.use('Agg')  # Use non-graphical backend

import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

print('Initializing')

# Define the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create the model instance
model = MLP()

# Create a dictionary mapping labels to help messages and tutorials
help_messages = {
    0: "Help message for label 0",
    1: "Help message for label 1",
    2: "Help message for label 2",
    3: "Help message for label 3",
    4: "Help message for label 4",
    5: "Help message for label 5",
    6: "Help message for label 6",
    7: "Help message for label 7",
    8: "Help message for label 8",
    9: "Help message for label 9",
}

tutorials = {
    0: "Tutorial for label 0",
    1: "Tutorial for label 1",
    2: "Tutorial for label 2",
    3: "Tutorial for label 3",
    4: "Tutorial for label 4",
    5: "Tutorial for label 5",
    6: "Tutorial for label 6",
    7: "Tutorial for label 7",
    8: "Tutorial for label 8",
    9: "Tutorial for label 9",
}

# Define the function to save the image and display help message
def save_image_and_help(image, label):
    image_pil = Image.fromarray(image)
    image_pil.save("image.png")
    messagebox.showinfo("Help", help_messages[label])

def display_tutorial(label):
    messagebox.showinfo("Tutorial", tutorials[label])

# Function to connect to real instrument
import cv2
instrument = None
inst_ix = 1
def connect_to_instrument():
    global instrument
    if instrument is not None:
        messagebox.showinfo("Connect", "Already connected")
        return
    instrument = cv2.VideoCapture(inst_ix)
    messagebox.showinfo("Connect", "Connected to real instrument")

# Function to disconnect from instrument
def disconnect_instrument():
    global instrument
    if instrument is not None:
        instrument.release()
        instrument = None
        messagebox.showinfo("Disconnect", "Successfully disconnected")
        return
    messagebox.showerror("Disconnect", "No instrument connected")

# Function to capture data
image = None
def capture_data():
    global image
    global instrument
    if instrument is None:
        messagebox.showerror("Capture", "No instrument connected")
        return
    success, image = instrument.read()
    if not success:
        messagebox.showerror("Capture", "Data capture failed")
        return
    image = image[...,::-1]
    print(image.shape)
    messagebox.showinfo("Capture", "Captured Image")

# Function to view data
def view_data():
    global image
    cv2.imshow("Captured Image", image[...,::-1])

# Function to load data
def load_data():
    global X_train, y_train, X_test, y_test
    url = "https://zenodo.org/record/6496656/files/dermamnist.npz"
    response = requests.get(url)
    data = np.load(BytesIO(response.content))
    X_train, y_train, X_test, y_test = data['train_images'], data['train_labels'], data['test_images'], data['test_labels']
    messagebox.showinfo("Load", "Data loaded")

# Function to train 1 epoch
def train(model, trainloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        optimizer.zero_grad()

        # Reshape the input tensor
        inputs = inputs.view(inputs.size(0), -1)

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
            inputs = inputs.view(inputs.size(0), -1)
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

# Function to train model
def train_model():
    global model
    global train_losses
    global train_accuracy
    # Step 8: Train Neural Network
    train_labels = torch.from_numpy(y_train.squeeze()).long()
    trainset = TensorDataset(torch.from_numpy(X_train).float(), train_labels)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

# Train the model
    train_losses = []
    train_accuracy = []
    num_epochs = 10
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train(model, trainloader, optimizer, criterion)
        acc = evaluate(model, trainloader, criterion)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {acc:.4f}")
    messagebox.showinfo("Train", f"Model trained: Train Loss: {train_losses[-1]:.4f}, Train accuracy: {acc:.4f}")


# Create the GUI
root = tk.Tk()
root.title("Microscope Dashboard")

# Create buttons
button_connect = tk.Button(root, text="Connect to Instrument", command=connect_to_instrument)
button_capture = tk.Button(root, text="Capture Data", command=capture_data)
button_load = tk.Button(root, text="Load Data", command=load_data)
button_train = tk.Button(root, text="Train Model", command=train_model)
#button_save = tk.Button(root, text="Save Model", command=save_model)
#button_test = tk.Button(root, text="Test Model", command=test_model)
#button_run = tk.Button(root, text="Run Model on New Image", command=run_model_on_new_image)
button_view = tk.Button(root, text="View Data", command=view_data)
#button_label = tk.Button(root, text="Label Data", command=label_data)

# Layout buttons
button_connect.pack()
button_capture.pack()
button_load.pack()
button_train.pack()
#button_save.pack()
#button_test.pack()
#button_run.pack()
button_view.pack()
#button_label.pack()

# Run the GUI
try:
    root.mainloop()
finally:
    # cleanup
    if instrument is not None:
        instrument.release()
