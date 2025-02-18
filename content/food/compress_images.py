from PIL import Image
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


def main():
    parser = argparse.ArgumentParser(
        description="Script to compress images and write to gallery.",
    )
    parser.add_argument("img_folder_path", type=str)
    parser.add_argument("site_path", type=str)
    parser.add_argument("mapping", type=str)

    args = parser.parse_args()

    # convert images
    for file in os.listdir(args.img_folder_path):
        if file.endswith(".jpg"):

            image = Image.open(os.path.join(args.img_folder_path, file))
            image.convert("RGB")
            image = crop_max_square(image)
            image = image.resize((750, 750))
            os.path.join(args.site_path, file)
            image.save(
                os.path.join(args.site_path, file).rstrip(".jpg")+".webp",
                "WEBP", quality=75, optimize=True
            )
    
    with open(args.mapping) as json_file:
        data = json.load(json_file)

    replacement_text = "\n".join([
        "![%s](./%s)" % (v, k)
        for k, v in data.items()
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
