from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")

emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def analyze_english_emotion(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        emotion_idx = int(torch.argmax(probs, dim=1).item())

    predicted_emotion = emotion_labels[emotion_idx]

    return {
        "text": text,
        "emotion": predicted_emotion,
        "probability": probs[0][emotion_idx].item(),
        "all_probabilities": dict(zip(emotion_labels, probs[0].tolist()))
    }

# Пример использования
if __name__ == "__main__":
    sample_text = "I hate you!"
    result = analyze_english_emotion(sample_text)
    print(result)
