from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
import argparse
import os
import json

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

def get_image_datetime(img):
    exif_data = img._getexif()
    if exif_data:
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == "DateTimeOriginal":  # Date the photo was taken
                value = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                return value    
    return None  # If no EXIF data found

def main():
    parser = argparse.ArgumentParser(
        description="Script to compress images and write to gallery.",
    )
    parser.add_argument("img_folder_path", type=str)
    parser.add_argument("site_path", type=str)
    parser.add_argument("mapping", type=str)

    args = parser.parse_args()

    with open(args.mapping) as json_file:
        data = json.load(json_file)

    

    files = []
    for file in os.listdir(args.img_folder_path):
        if file.endswith(".jpg"):
            image = Image.open(os.path.join(args.img_folder_path, file))
            date_of_img = get_image_datetime(image)
            files.append((file, date_of_img))
    files.sort(key=lambda x: x[1])

    filename_to_text = {}
    # convert images
    c = 1

    for file, _ in files:
        if file.endswith(".jpg"):

            image = Image.open(os.path.join(args.img_folder_path, file))
            date_of_img = get_image_datetime(image).strftime("%d/%m/%Y")
            if date_of_img is None:
                print("No date found in %s" % file)
            image.convert("RGB")
            image = crop_max_square(image)
            image = image.resize((750, 750))
            save_path = os.path.join(args.site_path, "%s.webp" % c)
            image.save(
                save_path,
                "WEBP", quality=75, optimize=True
            )
            text = data[date_of_img]
            filename_to_text["./%s.webp" % c] = [text + " (%s)" % date_of_img, date_of_img]
            c+=1

    
    replacement_text = "\n".join([
        "![%s](%s)" % (v[0], k)
        for k, v in sorted(
            filename_to_text.items(),
            key=lambda p: datetime.strptime(p[1][1], "%d/%m/%Y"),
            reverse=False
            )
    ])
    with open(os.path.join(args.site_path, "food_backup.md"), "r") as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        if line.startswith("INSERT"):
            modified_lines.append(replacement_text + "\n")  # Replace the line
        else:
            modified_lines.append(line)  # Keep the line unchanged

    # Write the modified content back to the file
    with open(os.path.join(args.site_path, "food.md"), "w") as file:
        file.writelines(modified_lines)


if __name__ == "__main__":
    main()
