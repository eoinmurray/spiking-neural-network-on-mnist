from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

def load_mnist_data(batch_size=128, dataset='MNIST', subset_size=None):
    """
    Load MNIST or FashionMNIST, optionally returning only the first `subset_size` examples.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    
    # choose the dataset class
    if dataset == 'MNIST':
        DS = MNIST
    elif dataset == 'FashionMNIST':
        DS = FashionMNIST
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # download / load full datasets
    train_ds = DS(root=".", train=True,  download=True, transform=transform)
    test_ds  = DS(root=".", train=False, download=True, transform=transform)

    # if subset_size is specified, take only the first subset_size examples
    if subset_size is not None:
        train_ds = Subset(train_ds, indices=list(range(subset_size)))
        test_ds  = Subset(test_ds,  indices=list(range(subset_size)))

    # wrap in DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
