from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# load tokenizer and model from huggingface
tokenizer = AutoTokenizer.from_pretrained("koshin2001/Japanese-to-emotions")
model = AutoModelForSequenceClassification.from_pretrained("koshin2001/Japanese-to-emotions")

emotion_labels = [
    "amusement", "anger", "excitement", "fear",
    "gratitude", "joy", "pride", "relief"
]

# map emotions to groups
group_map = {
    "amusement": "positive",
    "excitement": "positive",
    "gratitude": "positive",
    "joy": "positive",
    "anger": "negative",
    "fear": "negative",
    "relief": "negative",
    "pride": "neutral"
}

def analyze_japanese_emotion(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    # run the model in inference mode
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        emotion_idx = int(torch.argmax(probs, dim=1).item())

    predicted_emotion = emotion_labels[emotion_idx]
    emotion_group = group_map.get(predicted_emotion, "neutral")

    # return a json type object
    return {
        "text": text,
        "emotion": predicted_emotion,
        "group": emotion_group,
        "probability": probs[0][emotion_idx].item(),
        "all_probabilities": dict(zip(emotion_labels, probs[0].tolist()))
    }

if __name__ == "__main__":
    # example usage
    japanese_text = "私はとても嬉しいです！"
    result = analyze_japanese_emotion(japanese_text)
    print(result)
