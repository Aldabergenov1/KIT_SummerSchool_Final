# library for audio transcription, link: https://github.com/openai/whisper
import whisper
from typing import Any, Union


def audio_transcription(audio_file: str, model_name: str = "small") -> Union[str, list[Any]]:
    """
    Function to transcribe audio files on japanese using OpenAI's Whisper model.
    Args:
        audio_file (str): Path to the audio file.
        model_name (str): The name of the Whisper model to use. Default is "small"
    Returns:
        str: Transcribed text in Japanese.
    """
    # load the Whisper model
    model = whisper.load_model(model_name)

    # transcribe an audio file
    result = model.transcribe(audio_file, language='ja')

    # extract the Japanese text from the result
    japanese_text = result["text"]
    
    return japanese_text


if __name__ == "__main__":
    # example usage
    audio_file = "/home/makharon/Documents/Python/KIT_summerSchool/test_files/JAPFND1_0052.mp3"
    transcription = audio_transcription(audio_file=audio_file)
    print(transcription)
