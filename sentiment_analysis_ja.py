from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")

# model output logits correspond to the following emotions:
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


def analyze_english_emotion(text: str) -> dict:
    """
    Analyze the emotional tone of an English sentence.
    
    Args:
        text (str): The input sentence to analyze.

    Returns:
        dict: JSON response.
    """
    
    # tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    
    # inference mode
    with torch.no_grad():
        outputs = model(**inputs) # forward pass
        probs = F.softmax(outputs.logits, dim=1) # logits to probabilities
        emotion_idx = int(torch.argmax(probs, dim=1).item()) 

    # most probable emotion
    predicted_emotion = emotion_labels[emotion_idx]

    # form json
    return {
        "text": text,
        "emotion": predicted_emotion,
        "probability": probs[0][emotion_idx].item(),
        "all_probabilities": dict(zip(emotion_labels, probs[0].tolist()))
    }

# example usage
if __name__ == "__main__":
    sample_text = "Today the weather is so bad"
    result = analyze_english_emotion(sample_text)
    print(result)
