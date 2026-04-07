import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
import numpy as np
import json
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
import random

# Load the saved processor and model
model_save_path = "/work/kaippilr/food_ingredient_detection/model_blip_instructions_200"
processor = BlipProcessor.from_pretrained(model_save_path)
model = BlipForConditionalGeneration.from_pretrained(model_save_path)

# Load pre-trained BERT model for embeddings
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()

def cosine_similarity_score(predicted_instructions, actual_instructions):
    pred_embeddings = np.mean([get_bert_embeddings(instr)[0] for instr in predicted_instructions], axis=0)
    actual_embeddings = np.mean([get_bert_embeddings(instr)[0] for instr in actual_instructions], axis=0)
    similarity = cosine_similarity([pred_embeddings], [actual_embeddings])[0][0]
    return similarity

def predict_instructions(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64, num_beams=5, early_stopping=True)
    predicted_instructions = processor.decode(outputs[0], skip_special_tokens=True)
    predicted_instructions_list = [instr.strip() for instr in predicted_instructions.split(',')]
    return predicted_instructions_list

# Directory containing test images and the annotations file
test_image_dir = '/work/kaippilr/food_ingredient_detection/dataset_for_blip_instruction_testing_200/images'
annotations_file = '/work/kaippilr/food_ingredient_detection/dataset_for_blip_instruction_testing_200/annotations.json'

# Load the annotations from JSON file
with open(annotations_file, 'r') as f:
    test_instructions = json.load(f)

# Initialize lists to store evaluation metrics
similarity_scores = []
true_labels = []
predicted_labels = []

# Evaluate the model on the entire test dataset
for item in test_instructions:
    image_name = item["image"]
    actual_instructions = eval(item["instructions"])  # Convert string representation of list to list
    image_path = os.path.join(test_image_dir, image_name)
    predicted_instructions = predict_instructions(image_path)
    
    # Compute cosine similarity score
    score = cosine_similarity_score(predicted_instructions, actual_instructions)
    similarity_scores.append(score)
    
    # Prepare for precision-recall calculations
    all_instructions = list(set(predicted_instructions + actual_instructions))
    true_vector = [1 if instr in actual_instructions else 0 for instr in all_instructions]
    pred_vector = [1 if instr in predicted_instructions else 0 for instr in all_instructions]
    
    true_labels.extend(true_vector)
    predicted_labels.extend(pred_vector)

# Calculate average cosine similarity score
average_similarity = np.mean(similarity_scores)
print(f"Average Cosine Similarity Score: {average_similarity}")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(true_labels, predicted_labels)
average_precision = average_precision_score(true_labels, predicted_labels)

# Save Precision-Recall Curve as a figure
plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (Average Precision: {average_precision:.2f})')
plt.savefig('/work/kaippilr/food_ingredient_detection/dataset_for_blip_instruction_testing_200/precision_recall_curve1.png')  # Save the plot

# Embedding Visualization
all_instructions = list(set(predicted_instructions + actual_instructions))
embeddings = np.array([get_bert_embeddings(instr)[0] for instr in all_instructions])

# Determine appropriate perplexity
n_samples = len(all_instructions)
perplexity_value = min(30, n_samples - 1)  # Use a value less than n_samples

# Reduce dimensionality
tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Save Embedding Visualization as a figure
plt.figure(figsize=(10, 7))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', label='Instructions')
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.title('Instruction Embeddings Visualization')
plt.legend()
plt.savefig('/work/kaippilr/food_ingredient_detection/dataset_for_blip_instruction_testing_200/embedding_visualization1.png')  # Save the plot

# Randomly select 5 samples from the test set for detailed inspection
random_samples = random.sample(test_instructions, 5)

# Print the predicted and actual instructions for each random sample
print("\nDetailed Inspection of 5 Random Samples:")
for item in random_samples:
    image_name = item["image"]
    actual_instructions = eval(item["instructions"])  # Convert string representation of list to list
    image_path = os.path.join(test_image_dir, image_name)
    predicted_instructions = predict_instructions(image_path)
    
    # Compute cosine similarity score for the sample
    score = cosine_similarity_score(predicted_instructions, actual_instructions)
    
    # Print the details
    print(f"\nImage: {image_name}")
    print(f"Actual Instructions: {actual_instructions}")
    print(f"Predicted Instructions: {predicted_instructions}")
    print(f"Cosine Similarity Score: {score:.4f}")
    print("-" * 50)

