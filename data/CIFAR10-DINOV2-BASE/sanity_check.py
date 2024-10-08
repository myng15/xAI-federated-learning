import os
import pickle
import random
import numpy as np

RAW_DATA_PATH = "./data/CIFAR10-DINOV2-BASE/"
DATA_PATH = "./all_data/"
TRAIN_PATH = os.path.join(DATA_PATH, "train")
TEST_PATH = os.path.join(DATA_PATH, "test")


def load_embeddings():
    # Load CIFAR-10 embeddings and labels from the .npz files
    train_data = np.load(os.path.join(RAW_DATA_PATH, 'train.npz'))
    test_data = np.load(os.path.join(RAW_DATA_PATH, 'test.npz'))

    train_filenames = train_data['filenames']
    train_embeddings = train_data['embeddings']
    train_labels = train_data['labels']
    test_filenames = test_data['filenames']
    test_embeddings = test_data['embeddings']
    test_labels = test_data['labels']

    return train_filenames, train_embeddings, train_labels, test_filenames, test_embeddings, test_labels


def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def sanity_check():
    # Load the original CIFAR-10 embeddings and labels
    train_filenames, train_embeddings, train_labels, test_filenames, test_embeddings, test_labels = load_embeddings()

    # Helper to extract embeddings/labels based on index
    def get_data(index):
        if index < 50_000:  # Assuming first 50k are train
            return train_filenames[index], train_embeddings[index], train_labels[index]
        else:
            return test_filenames[index - 50_000], test_embeddings[index - 50_000], test_labels[index - 50_000]

    # Sanity check for train and test datasets
    for mode, path in [('train', TRAIN_PATH), ('test', TEST_PATH)]:
        if not os.path.exists(path):
            print(f"No {mode} directory found. Skipping {mode} data.")
            continue
        
        print(f"\nSanity Check for {mode} data:")
        # Get all client folders and select 5 random clients
        client_folders = os.listdir(path)
        random_clients = random.sample(client_folders, min(5, len(client_folders)))

        #for client_folder in os.listdir(path):
        for client_folder in random_clients:
            client_path = os.path.join(path, client_folder)
            train_file = os.path.join(client_path, "train.pkl")
            test_file = os.path.join(client_path, "test.pkl")

            # Load train and test indices
            train_indices = load_data(train_file) if os.path.exists(train_file) else np.array([])
            test_indices = load_data(test_file) if os.path.exists(test_file) else np.array([])

            print(f"\nClient {client_folder}:")
            print(f" - Number of training samples: {len(train_indices)}")
            print(f" - Number of testing samples: {len(test_indices)}")

            # Pick a random train and test sample for verification
            if train_indices.size > 0:
                random_train_idx = random.choice(train_indices)
                train_filename, train_embedding, train_label = get_data(random_train_idx)
                print(f"   Random Train Sample: Index: {random_train_idx}, Embedding Shape: {train_embedding.shape}, Filename: {train_filename}, Label: {train_label}")

            if test_indices.size > 0:
                random_test_idx = random.choice(test_indices)
                test_filename, test_embedding, test_label = get_data(random_test_idx)
                print(f"   Random Test Sample: Index: {random_test_idx}, Embedding Shape: {test_embedding.shape}, Filename: {test_filename}, Label: {test_label}")


if __name__ == "__main__":
    sanity_check()
