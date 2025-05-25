import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from cnn_model import CNN_PyTorch
from dataset import get_pytorch_loaders

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf

def train_pytorch_model(num_epochs=10, save_path="model_pytorch.torch"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_PyTorch(num_classes=4).to(device)
    train_loader, test_loader = get_pytorch_loaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accuracies = []


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"[PyTorch] Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    torch.save(model.state_dict(), save_path)

    plt.plot(train_losses, label='Loss (PyTorch)', color='red')
    plt.plot(train_accuracies, label='Accuracy (PyTorch)', color='green')
    plt.title("PyTorch Training Loss & Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()

def create_tensorflow_model(input_shape=(224, 224, 3), num_classes=4):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.8),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_tensorflow_model(train_generator, test_generator, num_epochs=10, save_path="model_tf.keras"):
    model = create_tensorflow_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_generator, epochs=num_epochs, validation_data=test_generator)
    model.save(save_path)

    plt.plot(history.history['loss'], label='Loss (TF)', color='red')
    plt.plot(history.history['accuracy'], label='Accuracy (TF)', color='green')
    plt.title("TensorFlow Training Loss & Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()
