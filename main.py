from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pathlib import Path
import shutil
import tempfile

from starlette.requests import Request

import time
import json
import asyncio

from japanese_ocr import japanese_ocr
from ja_to_end_translation import translate_japanese_to_english
from sentiment_analysis_ja import analyze_english_emotion
from voice_recognition import audio_transcription

app = FastAPI()


RESULT_PATH = Path(__file__).resolve().parent / "esp_result.json"
print("Saving to absolute path:", RESULT_PATH.resolve())


def save_result_to_file(result: dict):
    result["retrieved"] = False
    try:
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        print("Saved result to file.")
    except Exception as e:
        print(f"Error saving file: {e}")



def load_result_from_file():
    if RESULT_PATH.exists():
        try:
            with open(RESULT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading file: {e}")
    return None


def mark_result_as_retrieved():
    if RESULT_PATH.exists():
        try:
            with open(RESULT_PATH, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data["retrieved"] = True
                f.seek(0)
                json.dump(data, f, ensure_ascii=False)
                f.truncate()
        except Exception as e:
            print("Failed to mark as retrieved:", e)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def main_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.heic', '.heif')):
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
    emotion_result = analyze_english_emotion(english_text)

    result =  {
        "source": "image",
        "japanese_text": japanese_text,
        "english_translation": english_text,
        "emotion_analysis": emotion_result
    }
    
    save_result_to_file(result)
    print("Saving result:", result)
    save_result_to_file(result)

    return result


@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a', '.mov')):
        raise HTTPException(status_code=400, detail="Unsupported audio file type")

    suffix = Path(file.filename).suffix if file.filename else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name
    file.file.close()

    try:
        japanese_text = audio_transcription(temp_path)
        if not isinstance(japanese_text, str):
            return JSONResponse(status_code=500, content={"error": "OCR did not return a string"})
    except Exception as e:
        Path(temp_path).unlink(missing_ok=True)
        return JSONResponse(status_code=500, content={"error": f"Transcription failed: {str(e)}"})

    Path(temp_path).unlink(missing_ok=True)

    english_text = translate_japanese_to_english(japanese_text)
    emotion_result = analyze_english_emotion(english_text)

    result =  {
        "source": "audio",
        "japanese_text": japanese_text,
        "english_translation": english_text,
        "emotion_analysis": emotion_result
    }
    
    save_result_to_file(result)
    print("Saving result:", result)
    save_result_to_file(result)

    return result
    
    

@app.get("/esp_check/")
async def esp_check():
    result = load_result_from_file()
    if result and not result.get("retrieved", False):
        asyncio.create_task(delayed_mark_result_as_retrieved())
        return result
    return {"status": "no new data"}

async def delayed_mark_result_as_retrieved():
    await asyncio.sleep(2)
    mark_result_as_retrieved()
    print("Marked result as retrieved (after ESP pulled it)")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
