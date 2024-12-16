from fastapi import FastAPI, Form, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os

from app.keyframe_extraction import extract_keyframes
from app.image_captioning import generate_caption
from app.search import search_caption

app = FastAPI()

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Folder to save uploaded videos
UPLOAD_FOLDER = "data/videos/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route for the favicon.ico
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main upload and search page."""
    return templates.TemplateResponse("index.html", {"request": request, "prompt": "", "video_uploaded": False})


@app.post("/process", response_class=HTMLResponse)
async def process_video(request: Request, video: UploadFile = File(...), prompt: str = Form(...)):
    """Handle video upload and text prompt input."""
    try:
        # Save the uploaded video
        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        # Extract keyframes from video
        keyframes = extract_keyframes(video_path)

        if not keyframes:
            return templates.TemplateResponse("index.html", {"request": request, "error": "No keyframes were extracted from the video.", "prompt": prompt, "video_uploaded": True})

        # Generate captions for keyframes
        captions = [generate_caption(frame) for frame in keyframes]

        if not captions:
            return templates.TemplateResponse("index.html", {"request": request, "error": "No captions were generated for the keyframes.", "prompt": prompt, "video_uploaded": True})

        # Perform the search
        matching_caption = search_caption(prompt, captions)

        if not matching_caption:
            return templates.TemplateResponse("index.html", {"request": request, "error": "No matching caption found.", "prompt": prompt, "video_uploaded": True})

        return templates.TemplateResponse(
            "result.html",
            {"request": request, "prompt": prompt, "result": matching_caption}
        )
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"An error occurred: {str(e)}", "prompt": prompt, "video_uploaded": True})
