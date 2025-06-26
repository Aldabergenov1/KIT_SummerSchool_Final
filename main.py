from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import tempfile

from japanese_ocr import japanese_ocr
from ja_to_end_translation import translate_japanese_to_english
from sentiment_analysis_ja import analyze_japanese_emotion
from voice_recognition import audio_transcription

app = FastAPI()


@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        raise HTTPException(status_code=400, detail="Unsupported image file type")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name
    file.file.close()

    # OCR
    ocr_response = japanese_ocr(temp_path)
    Path(temp_path).unlink(missing_ok=True)

    if not isinstance(ocr_response, dict) or "data" not in ocr_response or not isinstance(ocr_response["data"], list):
        return JSONResponse(status_code=500, content={"error": "OCR failed or unexpected format"})

    japanese_text = ocr_response["data"][0]
    english_text = translate_japanese_to_english(japanese_text)
    emotion_result = analyze_japanese_emotion(japanese_text)

    return {
        "source": "image",
        "japanese_text": japanese_text,
        "english_translation": english_text,
        "emotion_analysis": emotion_result
    }


@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported audio file type")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name
    file.file.close()

    try:
        japanese_text = audio_transcription(temp_path)
    except Exception as e:
        Path(temp_path).unlink(missing_ok=True)
        return JSONResponse(status_code=500, content={"error": f"Transcription failed: {str(e)}"})

    Path(temp_path).unlink(missing_ok=True)

    english_text = translate_japanese_to_english(japanese_text)
    emotion_result = analyze_japanese_emotion(japanese_text)

    return {
        "source": "audio",
        "japanese_text": japanese_text,
        "english_translation": english_text,
        "emotion_analysis": emotion_result
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
