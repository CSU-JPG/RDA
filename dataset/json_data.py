import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from torchvision import transforms
import random
import yaml

class JsonImageDataset(Dataset):
    def __init__(self, json_paths, transform=None):
        """
        Args:
            json_paths (list[str]): list of json file paths
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        if isinstance(json_paths, str):
            try:
                json_paths = json.loads(json_paths)
            except Exception as e:
                raise ValueError(f"Failed to parse json_paths as list: {json_paths}") from e
        
        assert isinstance(json_paths, list), f"json_paths must be a list, got {type(json_paths)}"
        print(f"[JsonImageDataset] Loading {len(json_paths)} json files...")

        self.image_paths = []
        # import pdb; pdb.set_trace()
        for json_path in json_paths:

            if json_path.endswith(".yaml"):
                self.image_paths.extend(self.load_and_merge_jsons(json_path))
                continue
            with open(json_path, 'r') as f:
                data = json.load(f)
            self.image_paths.extend(self.extract_image_paths(data, json_path))

        print(f"[JsonImageDataset] Total images loaded: {len(self.image_paths)}")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def extract_image_paths(self, data, source):
        if not data:
            return []
        if isinstance(data[0], dict):
            try:
                return [item['image'] for item in data]
            except KeyError as e:
                raise KeyError(f"Missing 'image' key in {source}") from e
        return data

    def load_and_merge_jsons(self, yaml_file):
        with open(yaml_file, 'r') as f:
            meta_data = yaml.safe_load(f)["META"]

        merged_data = []
        for entry in meta_data:
            path = entry["path"]
            ratio = entry.get("ratio", 1.0)  # Default ratio is 1.0 (use all data)
            print(f"Loading data from {path} with ratio {ratio}")

            try:
                with open(path, 'r') as json_file:
                    json_data = json.load(json_file)
                    if ratio < 1.0:
                        sampled_data = random.sample(json_data, int(len(json_data) * ratio))
                        merged_data.extend(self.extract_image_paths(sampled_data, path))
                    else:
                        merged_data.extend(self.extract_image_paths(json_data, path))
            except Exception as e:
                print(f"Error loading data from {path}: {e}")
                continue
        print(f"Total number of samples: {len(merged_data)}")
        return merged_data

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"[JsonImageDataset] Failed to load image: {img_path}") from e

        if self.transform:
            img = self.transform(img)

        return img, img_path

def build_json_data(args, transform):
    return JsonImageDataset(args.data_path, transform=transform)