import argparse
from train import train_pytorch_model, train_tensorflow_model
from dataset import get_tensorflow_generators

def main():
    parser = argparse.ArgumentParser(description="Image Classification Trainer")
    parser.add_argument('--framework', type=str, choices=['pytorch', 'tensorflow'], required=True, help="Framework to use for training")
    args = parser.parse_args()

    if args.framework == 'pytorch':
        print("\n Lancement de l'entraînement PyTorch...")
        train_pytorch_model()

    elif args.framework == 'tensorflow':
        print("\n Lancement de l'entraînement TensorFlow...")
        train_generator, test_generator = get_tensorflow_generators()
        train_tensorflow_model(train_generator, test_generator)

if __name__ == '__main__':
    main()
