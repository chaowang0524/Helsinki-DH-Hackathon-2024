from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the pretrained model and processor
# set the "processer path" and "model path" before use
processor = AutoImageProcessor.from_pretrained("processer path")
model = AutoModel.from_pretrained("model path", output_attentions=True)

config = model.config
num_layers = config.num_hidden_layers
num_heads = config.num_attention_heads
print(f"The model has {num_layers} layers and {num_heads} heads per layer.")

# Load and process the image
image_path = "image to be processed"
save_path = "final output plotting"
image = Image.open(image_path).convert("RGB")
image = image.resize((512, 512))
inputs = processor(images=image, return_tensors="pt")

# Forward pass through the model
outputs = model(**inputs)

# Extract the attention weights
attentions = (
    outputs.attentions
)  # This will be a tuple of attention weights from all layers


def show_attention_map(
    attentions,
    image,
    layers_to_plot=[0, 13, 26, 39],
    heads_to_plot=[0, 8, 16, 23],
    save_path=save_path,
):
    fig, axs = plt.subplots(len(layers_to_plot), len(heads_to_plot), figsize=(20, 20))

    for row, layer in enumerate(layers_to_plot):
        for col, head in enumerate(heads_to_plot):
            # Get the attention map for the specified layer and head
            attention = attentions[layer][0, head].detach().cpu().numpy()
            print(attention.shape)
            # Exclude the classification token
            attention = attention[1:, 1:]

            # Calculate the number of patches per dimension
            num_patches = int(
                np.sqrt(attention.shape[0])
            )  # Should be 14 for a 14x14 grid

            assert (
                num_patches == 16
            ), f"Expected 16 patches per dimension, but got {num_patches}"

            # Reshape attention to (16, 16, 16, 16)
            attention_map = attention.reshape(
                num_patches, num_patches, num_patches, num_patches
            ).mean(axis=(2, 3))

            # Normalize the attention map
            attention_map = (attention_map - attention_map.min()) / (
                attention_map.max() - attention_map.min()
            )

            # Resize the attention map to match the image size
            attention_map = Image.fromarray(
                (attention_map * 512).astype(np.uint8)
            ).resize(image.size, resample=Image.BILINEAR)
            attention_map = np.array(attention_map) / 512.0

            # Overlay the attention map on the original image
            ax = axs[row, col]
            ax.imshow(image)
            im = ax.imshow(
                attention_map, cmap="hot", alpha=0.6
            )  # Adjust alpha for transparency
            ax.axis("off")
            ax.set_title(f"Layer {layer + 1}, Head {head + 1}")

    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.02, pad=0.1)
    cbar.set_label("Attention Score")
    plt.savefig(save_path)
    plt.show()


# Show the attention map and save the plot for the specified layers and heads
show_attention_map(
    attentions,
    image,
    layers_to_plot=[0, 13, 26, 39],
    heads_to_plot=[0, 8, 16, 23],
    save_path=save_path,
)
