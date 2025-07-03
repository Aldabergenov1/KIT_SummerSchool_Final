from googletrans import Translator

def translate_japanese_to_english(japanese_text: str) -> str:
    """
    Translate Japanese text to English using Google Translate API.

    Args:
        japanese_text (str): The input text in Japanese.

    Returns:
        str: The translated text in English.
    """
    
    translator = Translator()
    result = translator.translate(japanese_text, src='ja', dest='en')
    
    return result.text


if __name__ == "__main__":
    # Example usage
    japanese_text = "はじめまして どうぞよろしく"
    
    translated_text = translate_japanese_to_english(japanese_text)
    print(f"Original: {japanese_text}")
    print(f"Translated: {translated_text}")