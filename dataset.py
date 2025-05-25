import os

# Pour PyTorch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Pour TensorFlow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_pytorch_loaders(data_dir="/home/students/Documents/Computer vision/image_classification_app/data", batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "training"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "testing"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_tensorflow_generators(data_dir="data", batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_directory(
        os.path.join(data_dir, "training"),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_gen = datagen.flow_from_directory(
        os.path.join(data_dir, "testing"),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_gen, test_gen
