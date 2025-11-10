# Do not modify this file
import matplotlib.pyplot as plt
import numpy as np

seed = 42
np.random.seed(seed)
rng = np.random.RandomState(seed)


def visualize_image(image: np.ndarray, label: int, save_to: str = None):
    """Visualize a single image from the dataset.
    Args:
        image (np.ndarray): The image to visualize, expected shape (784,)
        label (int): The label of the image.
    """
    image_reshaped = image.reshape(28, 28)
    plt.imshow(image_reshaped, cmap="gray")
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()
    if save_to:
        plt.savefig(save_to)
    plt.close()
