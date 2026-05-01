from PIL import Image, ImageChops, ImageEnhance

def convert_to_ela(image_path, quality=90):
    """
    Enhanced ELA conversion with higher visibility for training.
    """
    original = Image.open(image_path).convert('RGB')
    
    # Temp JPEG save
    temp_path = "temp_ela_train.jpg"
    original.save(temp_path, 'JPEG', quality=quality)
    resaved = Image.open(temp_path)
    
    # Difference calculation
    ela_image = ImageChops.difference(original, resaved)
    
    # Increased scale for better detection of subtle edits
    scale = 20.0 
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image