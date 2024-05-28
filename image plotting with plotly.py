import plotly.io as pio
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import base64
from io import BytesIO
from tqdm import tqdm


def encode_image(image_path, image_size):
    """
    Encodes an image to a base64 string.

    :param image_path: Path to the image file.
    :return: Base64 encoded string of the image.
    """
    img = Image.open(image_path)
    img = img.resize(image_size)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


image_size = (256, 256)
image_folder = "your image directory"

import plotly.io as pio

fig = go.Figure()
# Add scatter trace for tooltips
fig.add_trace(
    go.Scatter(
        x=df["t-SNE1"],
        y=df["t-SNE2"],
        mode="markers",
        marker=dict(size=10, opacity=0.5),
        text=df["Image"],
        hoverinfo="text",
    )
)


for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
    x, y, image_name = row["t-SNE1"], row["t-SNE2"], row["Image"]
    image_path = f"{image_folder}/{image_name}"
    img_str = encode_image(image_path, image_size)

    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{img_str}",
            x=x,
            y=y,
            xref="x",
            yref="y",
            sizex=0.1,
            sizey=0.1,
            xanchor="center",
            yanchor="middle",
        )
    )

fig.update_layout(
    xaxis=dict(range=[df["t-SNE1"].min() - 1, df["t-SNE1"].max() + 1]),
    yaxis=dict(range=[df["t-SNE2"].min() - 1, df["t-SNE2"].max() + 1]),
    width=1600,
    height=1600,
    margin=dict(l=0, r=0, t=0, b=0),  # Remove padding
)

fig.show()
