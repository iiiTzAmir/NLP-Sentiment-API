import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F


MODEL_ID = "iiTzAmir21/distilbert-imdb-sentiment"


# Load dataset
dataset = load_dataset("stanfordnlp/imdb")

test_dataset = dataset["test"].shuffle(seed=42).select(range(500))


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

model.eval()


true_labels = []
predicted_labels = []
errors = []

all_confidences = []
all_correct = []


for example in test_dataset:
    text = example["text"] # type: ignore
    true_label = example["label"] # type: ignore

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        
        
    probabilities = torch.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()
    
    confidence = probabilities[0][prediction].item() # type: ignore
    
    all_confidences.append(confidence)
    all_correct.append(prediction == true_label)
    
    true_labels.append(true_label)
    predicted_labels.append(prediction)
    
    if prediction != true_label:
        errors.append({
            "text": text,
            "true_label": true_label,
            "predicted_label": prediction,
            "confidence": confidence
        })
    
print(f"\nTotal errors: {len(errors)}")

# Convert errors to DataFrame
errors_df = pd.DataFrame(errors)

errors_df["true_label"] = errors_df["true_label"].map({
    0: "Negative",
    1: "Positive"
})

errors_df["predicted_label"] = errors_df["predicted_label"].map({
    0: "Negative",
    1: "Positive"
})


# Sort by confidence
errors_df = errors_df.sort_values(
    by="confidence",
    ascending=False
)


# Most confident errors
print("\nMost confident errors:")

for index, row in errors_df.head(10).iterrows():
    print("\n--- Error ---")
    print("Text:", row["text"])
    print("True:", row["true_label"])
    print("Predicted:", row["predicted_label"])
    print(f"Confidence: {row['confidence']:.2%}")

# Temperature Scaling

logits_list = []
labels_list = []

for example in test_dataset:

    text = example["text"] # type: ignore
    true_label = example["label"] # type: ignore

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits_list.append(outputs.logits)
    labels_list.append(true_label)


logits = torch.cat(logits_list)
labels = torch.tensor(labels_list)


temperatures = torch.arange(
    0.5,
    3.01,
    0.05
)

best_temperature = 1.0
best_loss = float("inf")

for temperature in temperatures:

    scaled_logits = logits / temperature

    loss = F.cross_entropy(
        scaled_logits,
        labels
    )

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_temperature = temperature.item()


print("\nTemperature Scaling:")
print(f"Best temperature: {best_temperature:.2f}")
print(f"Best calibration loss: {best_loss:.4f}")


# Error statistics
print("\nError statistics:")

print(
    "Negative → Positive:",
    len(
        errors_df[
            (errors_df["true_label"] == "Negative") &
            (errors_df["predicted_label"] == "Positive")
        ]
    )
)

print(
    "Positive → Negative:",
    len(
        errors_df[
            (errors_df["true_label"] == "Positive") &
            (errors_df["predicted_label"] == "Negative")
        ]
    )
)

print(
    "Average confidence on errors:",
    f"{errors_df['confidence'].mean():.2%}"
)

# Calibration Analysis

print("\nCalibration Analysis:")

bins = [
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0)
]

for lower, upper in bins:

    bin_data = [
        correct
        for confidence, correct
        in zip(all_confidences, all_correct)
        if lower <= confidence < upper
    ]

    if len(bin_data) == 0:
        continue

    accuracy = sum(bin_data) / len(bin_data)

    print(
        f"{lower:.0%}-{upper:.0%} | "
        f"Samples: {len(bin_data)} | "
        f"Accuracy: {accuracy:.2%}"
    )

# Detailed High-Confidence Analysis

print("\nDetailed High-Confidence Analysis:")

high_confidence_bins = [
    (0.90, 0.92),
    (0.92, 0.94),
    (0.94, 0.96),
    (0.96, 0.98),
    (0.98, 1.00)
]

for lower, upper in high_confidence_bins:

    bin_data = [
        correct
        for confidence, correct
        in zip(all_confidences, all_correct)
        if lower <= confidence < upper
    ]

    if len(bin_data) == 0:
        print(
            f"{lower:.0%}-{upper:.0%} | "
            f"Samples: 0 | "
            f"Accuracy: N/A"
        )
        continue

    accuracy = sum(bin_data) / len(bin_data)

    print(
        f"{lower:.0%}-{upper:.0%} | "
        f"Samples: {len(bin_data)} | "
        f"Accuracy: {accuracy:.2%}"
    )

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)

print("Confusion Matrix:")
print(cm)


# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=["Negative", "Positive"]
    )
)


# Plot
plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks([0, 1], ["Negative", "Positive"])
plt.yticks([0, 1], ["Negative", "Positive"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()