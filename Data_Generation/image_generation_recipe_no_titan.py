import os
import shutil
import json

# For Stable Diffusion
from diffusers import StableDiffusionPipeline

# For Heatmap Generation
import daam

from PIL import Image
import torch
import gc
import warnings
import time

# Read the CSV file to get the prompts
import pandas as pd


from dotenv import load_dotenv
import os

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

load_dotenv()

# Suppress noisy torch.load FutureWarning from diffusers/transformers
warnings.filterwarnings(
    "ignore",
    message=r".*weights_only=False.*",
    category=FutureWarning,
)

CONFIG_PATH = os.getenv("CONFIG_PATH", os.path.join(os.path.dirname(__file__), "config.json"))
config = {}
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f) or {}
    except Exception:
        config = {}


def get_cfg(key, default=None):
    return os.getenv(key, config.get(key, default))


HF_TOKEN = get_cfg("HF_TOKEN")
MODEL_ID = get_cfg("MODEL_ID", "stabilityai/stable-diffusion-2-base")
LOCAL_ONLY = str(get_cfg("LOCAL_ONLY", "0")) == "1"
DISABLE_SAFETY = str(get_cfg("DISABLE_SAFETY", "1")) == "1"
GEN_SIZE = int(get_cfg("GEN_SIZE", "512"))
NUM_INFERENCE_STEPS = int(get_cfg("NUM_INFERENCE_STEPS", "30"))
NUM_IMAGES_PER_PROMPT = int(get_cfg("NUM_IMAGES_PER_PROMPT", "10"))
MAX_PROMPTS = int(get_cfg("MAX_PROMPTS", "200"))
print(f"[env] HF_TOKEN loaded: {'yes' if HF_TOKEN else 'no'}")
print(f"[env] MODEL_ID: {MODEL_ID}")
print(f"[env] LOCAL_ONLY: {LOCAL_ONLY}")
print(f"[env] DISABLE_SAFETY: {DISABLE_SAFETY}")
print(f"[env] GEN_SIZE: {GEN_SIZE}")
print(f"[env] NUM_INFERENCE_STEPS: {NUM_INFERENCE_STEPS}")
print(f"[env] NUM_IMAGES_PER_PROMPT: {NUM_IMAGES_PER_PROMPT}")
print(f"[env] MAX_PROMPTS: {MAX_PROMPTS}")


def safe_dirname(value, fallback="prompt"):
    # Replace Windows-invalid filename characters and trim.
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if c in invalid else c for c in str(value)).strip()
    return cleaned or fallback



class PromptHandlerLite:
    """
    Lightweight stand-in for TITAN PromptHandler.
    Keeps the same return shape: list of (prompt, objects, metadata).
    """

    def clean_prompt(self, prompts):
        processed = []
        for p in prompts:
            if p is None:
                continue
            prompt = str(p).strip()
            if not prompt:
                continue
            # Naive object list: split on commas, strip whitespace.
            objects = [part.strip() for part in prompt.split(",") if part.strip()]
            processed.append((prompt, objects, None))
        return processed


class SimpleDataset:
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
        Saves:
        - basic bounding box (from heatmap)
        - caption (prompt)
        """
        prompt, _, _ = processed_prompt
        
        print(f"[annotate] Annotating image {filename} for prompt: {prompt}")
        print(f"[annotate] Heatmap type: {type(heatmap)}, shape: {getattr(heatmap, 'shape', 'N/A')}")

        # Convert heatmap → numpy
        try:
            heat = heatmap.numpy()
            print(f"[annotate] Heatmap converted to numpy: shape {heat.shape}")
        except Exception as e:
            heat = None
            print(f"[annotate] Failed to convert heatmap to numpy: {e}")

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
            "prompt": prompt,
        }

        # Caption JSON
        caption = {
            "file_name": filename,
            "caption": prompt,
        }

        self.annotations.append(annotation)
        self.captions.append(caption)
        self.images.append(filename)

    def save(self):
        """Save annotations + captions to disk"""

        # Save annotations
        if self.annotations:
            ann_path = os.path.join(self.annotation_dir, "annotations.json")
            existing = []
            if os.path.exists(ann_path):
                try:
                    with open(ann_path, "r") as f:
                        existing = json.load(f) or []
                except Exception:
                    existing = []
            with open(ann_path, "w") as f:
                json.dump(existing + self.annotations, f, indent=4)

        # Save captions
        if self.captions:
            cap_path = os.path.join(self.caption_dir, "captions.json")
            existing = []
            if os.path.exists(cap_path):
                try:
                    with open(cap_path, "r") as f:
                        existing = json.load(f) or []
                except Exception:
                    existing = []
            with open(cap_path, "w") as f:
                json.dump(existing + self.captions, f, indent=4)

    def clear(self):
        """Clear memory (like batching)"""
        self.annotations = []
        self.captions = []
        self.images = []


# Load prompts from the CSV file
csv_file_path = "/root/Data_Generation/recipe_ingredient_pair_.csv"
if not os.path.exists(csv_file_path):
    csv_file_path = os.path.join(os.path.dirname(__file__), "recipe_ingredient_pair_.csv")
print(f"[data] Loading CSV: {csv_file_path}")
df = pd.read_csv(csv_file_path)

# Assuming the prompts are in the first column
prompts = df.iloc[:, 0].tolist()
print(f"[data] Total prompts in CSV: {len(prompts)}")

# Take the first 200 elements to test
prompts = prompts[:MAX_PROMPTS]
print(f"[data] Prompts used for generation: {len(prompts)}")

# Load lightweight PromptHandler
prompt_handler = PromptHandlerLite()

# Filter out the objects from the prompts to be used for annotations
processed_prompts = prompt_handler.clean_prompt(prompts)
print(f"[data] Processed prompts: {len(processed_prompts)}")

# Diffusion Model Setup
DIFFUSION_MODEL_PATH = MODEL_ID
ALLOW_CPU = os.getenv("ALLOW_CPU", "0") == "1"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif ALLOW_CPU:
    DEVICE = "cpu"
    print("[warn] CUDA not available. Falling back to CPU because ALLOW_CPU=1.")
else:
    raise SystemExit(
        "[error] CUDA GPU not available. Set ALLOW_CPU=1 in .env to run on CPU."
    )
print(f"[device] Using device: {DEVICE}")
SAVE_AFTER_NUM_IMAGES = 1  # Number of images after which the annotation and caption files will be saved
TARGET_SIZE = (224, 224)  # Desired size for the generated images

# Load Model
if not HF_TOKEN and not LOCAL_ONLY:
    print(
        "[error] HF_TOKEN is not set. If the model is gated (e.g., Stable Diffusion 2), "
        "set HF_TOKEN or set LOCAL_ONLY=1 with a cached/local model path."
    )
    raise SystemExit(1)

print(f"[model] Loading diffusion model: {DIFFUSION_MODEL_PATH}")
load_start = time.time()
model = StableDiffusionPipeline.from_pretrained(
    DIFFUSION_MODEL_PATH,
    use_auth_token=HF_TOKEN if HF_TOKEN else None,
    local_files_only=LOCAL_ONLY,
    torch_dtype=torch.float16 if DEVICE == "cuda" else None,
)
print(f"[model] from_pretrained finished in {time.time() - load_start:.1f}s")
move_start = time.time()
model = model.to(DEVICE)  # Set it to something else if needed, make sure DAAM supports that
print(f"[model] moved to device in {time.time() - move_start:.1f}s")
print(f"[model] Model loaded on device: {DEVICE}")
if DISABLE_SAFETY:
    model.safety_checker = None
    model.requires_safety_checker = False
    print("[model] Safety checker disabled to reduce memory use.")

# Memory helpers
if hasattr(model, "enable_attention_slicing"):
    model.enable_attention_slicing()
if hasattr(model, "enable_vae_slicing"):
    model.enable_vae_slicing()
if hasattr(model, "enable_xformers_memory_efficient_attention"):
    try:
        model.enable_xformers_memory_efficient_attention()
        print("[model] Enabled xFormers memory efficient attention.")
    except Exception:
        pass

# The Dataset
dataset = SimpleDataset()
print(f"[dataset] Writing images to: {dataset.image_dir}")
print(f"[dataset] Writing annotations to: {dataset.annotation_dir}")
print(f"[dataset] Writing captions to: {dataset.caption_dir}")


def iter_with_progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)

# Generating and Annotating Generated Images
try:
    # Iterating over the processed_prompts
    outer_iter = iter_with_progress(
        enumerate(processed_prompts),
        total=len(processed_prompts),
        desc="Prompts",
        unit="prompt",
    )
    for i, processed_prompt in outer_iter:
        # Generating images for these processed prompts and annotating them
        inner_iter = iter_with_progress(
            range(NUM_IMAGES_PER_PROMPT),
            total=NUM_IMAGES_PER_PROMPT,
            desc=f"Images for prompt {i + 1}",
            unit="img",
            leave=False,
        )
        for j in inner_iter:
            # traversing the processed prompts
            prompt, _, _ = processed_prompt

            # generating images. keeping track of the attention heatmaps
            try:
                with daam.trace(model) as trc:
                    output_image = model(
                        prompt,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        height=GEN_SIZE,
                        width=GEN_SIZE,
                    ).images[0]
                    global_heat_map = trc.compute_global_heat_map()
                print(f"[daam] DAAM tracing completed successfully for prompt {i}, image {j}")
                print(f"[daam] Heatmap type: {type(global_heat_map)}, shape: {global_heat_map.shape if hasattr(global_heat_map, 'shape') else 'N/A'}")
            except RuntimeError as e:
                msg = str(e).lower()
                if "cuda" in msg or "out of memory" in msg or "memory allocation" in msg:
                    print(f"[warn] OOM at prompt {i}, image {j}. Skipping. Error: {e}")
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise

            # Resize the generated image
            resample = getattr(Image, "Resampling", Image).LANCZOS
            output_image = output_image.resize(TARGET_SIZE, resample)

            # Saving Generated Image
            output_image.save(os.path.join(dataset.image_dir, f"{i}_{j}.png"))

            # Object Annotate Generated Image using the attention heatmaps
            dataset.annotate(output_image, f"{i}_{j}.png", global_heat_map, processed_prompt)

            if len(dataset.images) % SAVE_AFTER_NUM_IMAGES == 0:
                # Saving Annotations on Disk
                dataset.save()
                # Freeing up Memory
                dataset.clear()

    if len(dataset.annotations):
        dataset.save()
        dataset.clear()

except KeyboardInterrupt:  # In case of KeyboardInterrupt save the annotations and captions
    dataset.save()
    dataset.clear()
    print("[warn] Interrupted. Partial annotations/captions saved.")

# Define the new base directory for the restructured folders
NEW_BASE_DIR = "New_Generated_Train"
NEW_IMAGES_DIR = os.path.join(NEW_BASE_DIR, "Images")
NEW_ANNOTATIONS_DIR = os.path.join(NEW_BASE_DIR, "Annotations")
NEW_CAPTIONS_DIR = os.path.join(NEW_BASE_DIR, "Captions")

# Create the new folder structure
os.makedirs(NEW_IMAGES_DIR, exist_ok=True)
os.makedirs(NEW_ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(NEW_CAPTIONS_DIR, exist_ok=True)
print(f"[export] New images dir: {NEW_IMAGES_DIR}")
print(f"[export] New annotations dir: {NEW_ANNOTATIONS_DIR}")
print(f"[export] New captions dir: {NEW_CAPTIONS_DIR}")


# Function to copy files to the new folder structure with renamed folders
def copy_files_to_new_structure(original_dir, new_dir, prefix, file_extension, num_per_prompt, prompts_list):
    file_counter = 1
    for prompt in prompts_list:
        prompt_dir = os.path.join(new_dir, safe_dirname(prompt))
        os.makedirs(prompt_dir, exist_ok=True)
        for j in range(num_per_prompt):
            src_file = os.path.join(original_dir, f"{prefix}-{file_counter}{file_extension}")
            dst_file = os.path.join(prompt_dir, f"{prefix}{j + 1}{file_extension}")
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)
            file_counter += 1


# Copy images
print("[export] Copying images...")
for i, prompt in iter_with_progress(
    enumerate(prompts),
    total=len(prompts),
    desc="Copy images",
    unit="prompt",
):
    prompt_dir = os.path.join(NEW_IMAGES_DIR, safe_dirname(prompt))
    os.makedirs(prompt_dir, exist_ok=True)
    for j in range(NUM_IMAGES_PER_PROMPT):
        src_file = os.path.join(dataset.image_dir, f"{i}_{j}.png")
        dst_file = os.path.join(prompt_dir, f"image{j + 1}.png")
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)

# Copy annotations
print("[export] Copying annotations...")
copy_files_to_new_structure(
    dataset.annotation_dir,
    NEW_ANNOTATIONS_DIR,
    "object-detect",
    ".json",
    NUM_IMAGES_PER_PROMPT,
    prompts,
)

# Copy captions
print("[export] Copying captions...")
copy_files_to_new_structure(
    dataset.caption_dir,
    NEW_CAPTIONS_DIR,
    "object-caption",
    ".json",
    NUM_IMAGES_PER_PROMPT,
    prompts,
)
