from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    try:
        image = Image.fromarray(image).convert("RGB")  # Convert NumPy array to Image
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        raise Exception(f"Error generating caption: {str(e)}")
