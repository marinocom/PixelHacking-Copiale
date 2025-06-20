import os
import json

# Set the path to your Frankenstein dataset
frankenstein_dir = '/home/moliveros/Datasets/FaustBibleKantDataset'

splits = ['train', 'val', 'test']

for split in splits:
    img_dir = os.path.join(frankenstein_dir, split)
    json_path = os.path.join(frankenstein_dir, f'{split}.json')

    # Count images
    if os.path.exists(img_dir):
        img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        num_imgs = len(img_files)
    else:
        num_imgs = 0
    # Count JSON entries
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        num_json = len(data)
    else:
        num_json = 0
    print(f"{split}: {num_imgs} images, {num_json} JSON entries") 