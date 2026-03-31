import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, Trainer, TrainingArguments
from PIL import Image
from datasets import Dataset
import os
import shutil
import json

# Config loading
CONFIG_PATH = os.getenv("CONFIG_PATH", os.path.join(os.path.dirname(__file__), "config.json"))
config = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f) or {}


def get_cfg(key, default=None):
    return os.getenv(key, config.get(key, default))

MODEL_ID = get_cfg("MODEL_ID", "Salesforce/blip-image-captioning-base")
print(f"[config] MODEL_ID: {MODEL_ID}")

# Load the BLIP processor and model
print("[model] Loading BLIP processor...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
print("[model] Loading BLIP model...")
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)

# Load the annotations
ANNOTATIONS_PATH = get_cfg(
    "RECIPE_ANNOTATIONS_PATH",
    "/work/kaippilr/food_ingredient_detection/dataset_for_blip_recipe_200/annotations.json",
)
IMAGES_DIR = get_cfg(
    "RECIPE_IMAGES_DIR",
    "/work/kaippilr/food_ingredient_detection/dataset_for_blip_recipe_200/images",
)
TEXT_FIELD = get_cfg("RECIPE_TEXT_FIELD", "ingredients")
print(f"[data] Annotations: {ANNOTATIONS_PATH}")
print(f"[data] Images dir: {IMAGES_DIR}")
print(f"[data] Text field: {TEXT_FIELD}")

with open(ANNOTATIONS_PATH, "r") as f:
    annotations = json.load(f)
print(f"[data] Loaded annotations: {len(annotations)}")

# Prepare the dataset dictionary
data = {
    "image_path": [],
    "ingredients": []
}

# Convert annotations to dataset format
for item in annotations:
    image_path = os.path.join(IMAGES_DIR, item["image"])
    ingredients = item[TEXT_FIELD]
    data["image_path"].append(image_path)
    data["ingredients"].append(ingredients)

# Convert to HuggingFace Dataset
dataset = Dataset.from_dict(data)
print(f"[data] Dataset size: {len(dataset)}")

def tokenize_function(examples):
    # Load and preprocess images
    images = [Image.open(image_path).convert("RGB") for image_path in examples['image_path']]

    # Process images and text using BLIP processor
    #inputs = processor(images=images, text=examples['ingredients'], return_tensors='pt', padding=True)


    inputs = processor(
        images=images,
        text=examples["ingredients"],
        return_tensors="pt",
        padding="max_length",  # Pad sequences to max length
        truncation=True,  # Truncate sequences that are longer than max_length
        max_length=int(get_cfg("MAX_LENGTH", 64)),
    )

    # Setting labels for the decoder, which is the tokenized ingredient text
    inputs['labels'] = inputs.input_ids.clone()

    return inputs

# Apply the tokenize function to the dataset
print("[data] Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns
tokenized_dataset = tokenized_dataset.remove_columns(["image_path", "ingredients"])
print(f"[data] Tokenized dataset columns: {tokenized_dataset.column_names}")

# Define the training arguments
training_args = TrainingArguments(
    output_dir=get_cfg("OUTPUT_DIR", "./results"),
    evaluation_strategy=get_cfg("EVAL_STRATEGY", "epoch"),
    per_device_train_batch_size=int(get_cfg("TRAIN_BATCH_SIZE", 4)),
    per_device_eval_batch_size=int(get_cfg("EVAL_BATCH_SIZE", 4)),
    num_train_epochs=float(get_cfg("NUM_TRAIN_EPOCHS", 3)),
    save_steps=int(get_cfg("SAVE_STEPS", 10000)),
    save_total_limit=int(get_cfg("SAVE_TOTAL_LIMIT", 2)),
    remove_unused_columns=str(get_cfg("REMOVE_UNUSED_COLUMNS", "False")) == "True",
    push_to_hub=str(get_cfg("PUSH_TO_HUB", "False")) == "True",
)
print("[train] TrainingArguments initialized")

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)
print("[train] Trainer initialized")

# Train the model
print("[train] Starting training...")
trainer.train()
print("[train] Training complete")

# Save the trained model and processor
model_save_path = get_cfg(
    "RECIPE_MODEL_SAVE_PATH",
    "/work/kaippilr/food_ingredient_detection/model_blip_recipe_200",
)
print(f"[save] Saving model to: {model_save_path}")
model.save_pretrained(model_save_path)
processor.save_pretrained(model_save_path)
print("[save] Save complete")
