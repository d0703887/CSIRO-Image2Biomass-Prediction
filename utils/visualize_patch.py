import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision.transforms import Resize
from torchvision.io import read_image
import numpy as np

# 1. Setup constants based on your description
IMG_SIZE = (512, 512)  # The size you resize to
PATCH_SIZE = 16  # The patch size for the ViT/DinoV3


def visualize_patch_scale(image_path):
    """
    Loads an image, applies the CSIRO split/resize logic,
    and visualizes the 16x16 patch size.
    """

    # Load raw image (Simulating your dataset logic)
    # If you don't have a path handy, we can create a dummy image below
    try:
        img = read_image(image_path)
    except Exception as e:
        print(f"Could not load image at {image_path}, creating dummy noise image.")
        img = torch.randint(0, 255, (3, 1000, 2000), dtype=torch.uint8)

    # --- Step A: Split Image (Logic from CSIRODataset) ---
    _, _, w = img.shape
    center = w // 2
    # We only need to visualize one half (e.g., Left) to see the scale
    left_img = img[:, :, :center]

    # --- Step B: Resize (Logic from your transform) ---
    resize_transform = Resize(IMG_SIZE)
    processed_img = resize_transform(left_img)

    # Convert to numpy for plotting (H, W, C)
    img_np = processed_img.permute(1, 2, 0).numpy()

    # --- Step C: Visualization ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Display the image
    ax.imshow(img_np)

    # 1. Visualizing a Single Patch (Red Box)
    # Let's put a patch in the center so you can see it clearly
    center_y, center_x = IMG_SIZE[0] // 2, IMG_SIZE[1] // 2
    # Align to grid
    start_y = (center_y // PATCH_SIZE) * PATCH_SIZE
    start_x = (center_x // PATCH_SIZE) * PATCH_SIZE

    rect = patches.Rectangle(
        (start_x, start_y), PATCH_SIZE, PATCH_SIZE,
        linewidth=2, edgecolor='r', facecolor='none', label='Single 16x16 Patch'
    )
    ax.add_patch(rect)

    # 2. (Optional) Visualize the Grid to see density
    # Draws faint yellow lines to show how the image is divided
    grid_color = 'yellow'
    grid_alpha = 0.3

    # Draw vertical lines
    for x in range(0, IMG_SIZE[1], PATCH_SIZE):
        ax.axvline(x, color=grid_color, alpha=grid_alpha, linewidth=0.5)

    # Draw horizontal lines
    for y in range(0, IMG_SIZE[0], PATCH_SIZE):
        ax.axhline(y, color=grid_color, alpha=grid_alpha, linewidth=0.5)

    ax.set_title(f"Image Size: {IMG_SIZE} | Patch Size: {PATCH_SIZE}x{PATCH_SIZE}\n(Red box is one patch)")
    ax.legend(loc='upper right')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # --- Run the visualization ---
    # Replace 'path/to/your/image.jpg' with a real file from your data folder
    visualize_patch_scale('../data/CSIRO/train/ID4464212.jpg')