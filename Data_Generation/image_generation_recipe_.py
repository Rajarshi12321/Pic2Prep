import os
import shutil
import csv


# For Stable Diffusion
from diffusers import StableDiffusionPipeline

# For Heatmap Generation
import daam

# For TITAN workflow
from titan import *

from PIL import Image

# Read the CSV file to get the prompts
import pandas as pd


import os
import json

class TITANDataset:
    def __init__(self, base_dir="generated_dataset"):
        self.base_dir = base_dir
        
        # Directories
        self.image_dir = os.path.join(base_dir, "images")
        self.annotation_dir = os.path.join(base_dir, "annotations")
        self.caption_dir = os.path.join(base_dir, "captions")
        
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.annotation_dir, exist_ok=True)
        os.makedirs(self.caption_dir, exist_ok=True)

        # Internal storage
        self.annotations = []
        self.captions = []
        self.images = []

    def annotate(self, image, filename, heatmap, processed_prompt):
        """
        Replaces TITAN's annotation logic.
        Saves:
        - basic bounding box (from heatmap)
        - caption (prompt)
        """

        prompt, _, _ = processed_prompt

        # Convert heatmap → numpy
        try:
            heat = heatmap.numpy()
        except:
            heat = None

        bbox = None

        if heat is not None:
            import numpy as np

            # Normalize heatmap
            heat = heat / (heat.max() + 1e-8)

            # Threshold to get important region
            mask = heat > 0.5

            if mask.any():
                coords = np.argwhere(mask)
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)

                bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]

        # Annotation JSON
        annotation = {
            "file_name": filename,
            "bbox": bbox,
            "prompt": prompt
        }

        # Caption JSON
        caption = {
            "file_name": filename,
            "caption": prompt
        }

        self.annotations.append(annotation)
        self.captions.append(caption)
        self.images.append(filename)

    def save(self):
        """Save annotations + captions to disk"""

        # Save annotations
        if self.annotations:
            ann_path = os.path.join(self.annotation_dir, "annotations.json")
            with open(ann_path, "w") as f:
                json.dump(self.annotations, f, indent=4)

        # Save captions
        if self.captions:
            cap_path = os.path.join(self.caption_dir, "captions.json")
            with open(cap_path, "w") as f:
                json.dump(self.captions, f, indent=4)

    def clear(self):
        """Clear memory (like TITAN batching)"""
        self.annotations = []
        self.captions = []
        self.images = []
        
# Load prompts from the CSV file
csv_file_path = '/work/kaippilr/food_ingredient_detection/recipes_1000.csv'
df = pd.read_csv(csv_file_path)

# Assuming the prompts are in the first column
prompts = df.iloc[:, 0].tolist()

# Take the first 2 elements of that list just to test and see
prompts = prompts[:200]
# Load PromptHandler from TITAN
prompt_handler = PromptHandler()

# Filter out the objects from the prompts to be used for annotations
processed_prompts = prompt_handler.clean_prompt(prompts)

#print(processed_prompts)

# Diffusion Model Setup
DIFFUSION_MODEL_PATH = 'stabilityai/stable-diffusion-2-base'
DEVICE = 'cuda'  # device
NUM_IMAGES_PER_PROMPT = 50  # Number of images to be generated per prompt
NUM_INFERENCE_STEPS = 50  # Number of inference steps to the Diffusion Model
SAVE_AFTER_NUM_IMAGES = 1  # Number of images after which the annotation and caption files will be saved
TARGET_SIZE = (224, 224)  # Desired size for the generated images

# Load Model
model = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL_PATH)
model = model.to(DEVICE)  # Set it to something else if needed, make sure DAAM supports that

# The TITAN Dataset
titan_dataset = TITANDataset()

# Generating and Annotating Generated Images
try:
    # Iterating over the processed_prompts
    for i, processed_prompt in enumerate(processed_prompts):
        # Generating images for these processed prompts and annotating them
        for j in range(NUM_IMAGES_PER_PROMPT):
            # traversing the processed prompts
            prompt, _, _ = processed_prompt

            print()
            #print(f'Prompt No.: {i + 1}/{len(processed_prompts)}')
            #print(f'Image No.: {j + 1}/{NUM_IMAGES_PER_PROMPT}')
            #print('Generating Image...')

            # generating images. keeping track of the attention heatmaps
            with daam.trace(model) as trc:
                output_image = model(prompt, num_inference_steps=NUM_INFERENCE_STEPS).images[0]
                global_heat_map = trc.compute_global_heat_map()

            # Resize the generated image
            output_image = output_image.resize(TARGET_SIZE, Image.ANTIALIAS)

            # Saving Generated Image
            output_image.save(os.path.join(titan_dataset.image_dir, f'{i}_{j}.png'))
            #print(f'Saved Generated Image... {i}_{j}.png')

            # Object Annotate Generated Image using the attention heatmaps
            #print(f'Adding Annotation for {i}_{j}.png')
            titan_dataset.annotate(output_image, f'{i}_{j}.png', global_heat_map, processed_prompt)

            if len(titan_dataset.images) % SAVE_AFTER_NUM_IMAGES == 0:
                #print()
                # Saving Annotations on Disk
                titan_dataset.save()
                # Freeing up Memory
                titan_dataset.clear()

    if len(titan_dataset.annotations):
        titan_dataset.save()
        titan_dataset.clear()

except KeyboardInterrupt:  # In case of KeyboardInterrupt save the annotations and captions
    titan_dataset.save()
    titan_dataset.clear()

# # Merge annotation and caption files
# merge_annotation_files()
# merge_caption_files()

# Define the new base directory for the restructured folders
NEW_BASE_DIR = 'New_Generated_Train'
NEW_IMAGES_DIR = os.path.join(NEW_BASE_DIR, 'Images')
NEW_ANNOTATIONS_DIR = os.path.join(NEW_BASE_DIR, 'Annotations')
NEW_CAPTIONS_DIR = os.path.join(NEW_BASE_DIR, 'Captions')

# Create the new folder structure
os.makedirs(NEW_IMAGES_DIR, exist_ok=True)
os.makedirs(NEW_ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(NEW_CAPTIONS_DIR, exist_ok=True)

# Function to copy files to the new folder structure with renamed folders
def copy_files_to_new_structure(original_dir, new_dir, prefix, file_extension, num_per_prompt, prompts_list):
    file_counter = 1
    for prompt in prompts_list:
        prompt_dir = os.path.join(new_dir, f'{prompt}')
        os.makedirs(prompt_dir, exist_ok=True)
        for j in range(num_per_prompt):
            src_file = os.path.join(original_dir, f'{prefix}-{file_counter}{file_extension}')
            dst_file = os.path.join(prompt_dir, f'{prefix}{j + 1}{file_extension}')
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)
            file_counter += 1

# Copy images
for i, prompt in enumerate(prompts):
    prompt_dir = os.path.join(NEW_IMAGES_DIR, f'{prompt}')
    os.makedirs(prompt_dir, exist_ok=True)
    for j in range(NUM_IMAGES_PER_PROMPT):
        src_file = os.path.join(titan_dataset.image_dir, f'{i}_{j}.png')
        dst_file = os.path.join(prompt_dir, f'image{j + 1}.png')
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)

# Copy annotations
copy_files_to_new_structure(titan_dataset.annotation_dir, NEW_ANNOTATIONS_DIR, 'object-detect', '.json',
                            NUM_IMAGES_PER_PROMPT, prompts)

# Copy captions
copy_files_to_new_structure(titan_dataset.caption_dir, NEW_CAPTIONS_DIR, 'object-caption', '.json',
                            NUM_IMAGES_PER_PROMPT, prompts)

# Load the Visualizer
#titan_visualizer = TITANViz()

# Interactive Annotation Visualizer
#titan_visualizer.visualize_annotation(image_id=1)

