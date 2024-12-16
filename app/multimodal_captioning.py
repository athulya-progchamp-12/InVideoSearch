from models.audio_captioning_model import generate_audio_caption
from app.image_captioning import generate_caption

def multimodal_captioning(image, audio_clip):
    image_caption = generate_caption(image)
    audio_caption = generate_audio_caption(audio_clip)
    return f"Image: {image_caption}. Audio: {audio_caption}."
