from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle

model_path = "model path"
processor_path = "processor path"
image_dir = "the image directory"
feature_path = "the path to store the output feature"  # optional as the the program will output the python dictionary with its keys being the image name and values the image feature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    processor = AutoImageProcessor.from_pretrained(processor_path)
    model = AutoModel.from_pretrained(model_path).to(device)
except Exception as e:
    print(f"Error loading local models: {e}")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
    model = AutoModel.from_pretrained("facebook/dinov2-giant").to(device)
    processor.save_pretrained(processor_path)
    model.save_pretrained(model_path)


class ProcessedImageDataset(Dataset):
    def __init__(self, image_dir, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.image_files = [
            f
            for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        processed_image = self.processor(images=image, return_tensors="pt").to(device)
        return processed_image, self.image_files[idx]


dataset = ProcessedImageDataset(image_dir=image_dir, processor=processor)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)


features_dict = dict()
model.eval()

for processed_inputs, filenames in tqdm(data_loader, desc="Processing images"):
    with torch.no_grad():
        input_tensor = processed_inputs["pixel_values"].squeeze(1).to(device)
        outputs = model(input_tensor)
        # Extract the [CLS] token features
        cls_features = outputs.last_hidden_state[
            :, 0, :
        ]  # Shape: [batch_size, hidden_size]
        for i, feature in enumerate(cls_features):
            features_dict[filenames[i]] = (
                feature.cpu().numpy()
            )  # feature has shape [1536]

# Specify the path for the output file
output_path = "your path/feature_dict.pkl"

# Save the dictionary to a pickle file
with open(output_path, "wb") as handle:
    pickle.dump(features_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
