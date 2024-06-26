# Helsinki-DH-Hackathon-2024
This repository is to store the code that I contributed to the team Enlightening Illustrations: Analyzing the Role of Images in Enlightenment-Era Luxury Books during the DHH24 Hackathon. The instruction of the code is provided.

It covers the main functions that I used in the DHH24 including image feature extraction, ploting image with Plotly and ploting the attention map.

### Quick start:  

In terminal (bash):    

```
git clone https://github.com/chaowang0524/Helsinki-DH-Hackathon-2024.git
```

```
cd Helsinki-DH-Hackathon-2024
```
Set up the virtual enviroment:
```
python -m venv .venv
```
Activate the virtual enviroment, so that the pip packages will be installed in the venv:
```
source .venv/bin/activate
```
Install the dependencies:
```
pip install -r requirements.txt
```
## Programs:

### Image Feature Extraction:

This code snippet `image_feature_extraction.py` is using a pretrained model DINOv2 (https://huggingface.co/facebook/dinov2-giant) to extract image feature for downstream tasks. 

This program leverages the power of GPUs so running in the environment with GPU is recommended.

This program takes image files under directory `image_dir` , output a python dictionary in binary `features_dict` (with keys being the image id and the value the image feature) for further process.

### Image Plotting with Plotly in Python

The program `image_plotting_with_plotly.py` uses the Python Plotly library to directly plot image onto a 2D system. The program takes a pandas dataframe (in the code it is the variable `df`) which needs at least three columns: `t-SNE1`, `t-SNE2` and `Image` where the `t-SNE` is a technique to do dimension reduction so you can also use `PCA` if necessary. 

### Plotting the Attention Map:

The program `attention_map.py` plots the average value of the attention score of each attention head across all transformer layers when we are using vision transformer models. This code resizes the image to be processed into a 512*512 pixel image then output the plotting of the attention score of the head 1,9,17,24 (index 0, 8, 16, 23) on the layer 1, 14, 27, 40 (index 0, 13, 26, 39) onto the resized image. 

Please modify the number of the head and layers to align with your need and with the model (as DINOv2-giant has 40 transformer layers and 24 heads on each layer).

The final output are 16 plottings in a 4*4 grid showing which attention head is focusing on which part of the image. The brighter the area, the more attention it is receiving. The final plot is saved to the same directory.

