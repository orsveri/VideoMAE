import cv2
import numpy as np
from PIL import Image


def create_image_matrix(images, grid_size=(4, 4)):
    """
    Create a grid of images resized to half their original size.

    Args:
        images (list): List of PIL.Image objects.
        grid_size (tuple): Dimensions of the grid (rows, cols).

    Returns:
        PIL.Image: Combined image grid.
    """
    # Resize images to half of their original size
    resized_images = [img.resize((img.width // 2, img.height // 2)) for img in images]

    # Get dimensions of the grid
    rows, cols = grid_size
    assert len(resized_images) == rows * cols, "Number of images must match grid size"

    # Determine the size of the combined image
    img_width, img_height = resized_images[0].size
    grid_width = cols * img_width
    grid_height = rows * img_height

    # Create a blank canvas for the grid
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Paste each image into the grid
    for idx, img in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        grid_image.paste(img, (col * img_width, row * img_height))

    return grid_image