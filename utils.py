from PIL import Image, ImageOps
from keras.preprocessing.image import img_to_array, load_img
from keras.applications import mobilenet

def add_border(image, border_color, border_size):
    """
    Add a border around the given image.
    """
    return ImageOps.expand(image, border=border_size, fill=border_color)

def merge_images(base_folder, image_paths, output_name, target_size=(200, 200), border_color="white", border_size=5):
    images = []
    # Load, resize, and add border to images
    for path in image_paths:
        img = Image.open(base_folder + '/' +path +'.jpg')
        img = img.resize(target_size)
        img = add_border(img, border_color, border_size)
        images.append(img)

    # Create a blank image with a size sufficient to contain all the resized images
    result_width = target_size[0] * 2 + border_size * 2
    result_height = target_size[1] * 2 + border_size * 2
    result = Image.new("RGB", (result_width, result_height), border_color)

    # Paste resized images into the blank image
    for i in range(len(images)):
        x = i % 2
        y = i // 2
        result.paste(images[i], (x * target_size[0] + border_size, y * target_size[1] + border_size))

    # Save the resulting image
    result.save(output_name)


# Preprocess images and extract features
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = mobilenet.preprocess_input(img_array)
    return img_array
