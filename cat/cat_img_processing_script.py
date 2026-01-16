#!/usr/bin/env python3
"""
Image to WebP converter with metadata removal.
Converts images from a source folder to WebP format and strips all metadata.
"""

import argparse
from pathlib import Path
from PIL import Image, ImageOps


def convert_to_webp(input_folder, output_folder="./cat/pics/", quality=85, max_width=1920):
    """
    Convert images to WebP format and remove all metadata.

    Args:
        input_folder: Path to folder containing source images
        output_folder: Path to output folder (default: ./cat/pics/)
        quality: WebP quality setting 1-100 (default: 85)
        max_width: Maximum width in pixels (default: 1920)
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}

    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist")
        return

    # Process all images in the input folder
    converted_count = 0
    for image_file in sorted(input_path.iterdir()):
        if image_file.suffix.lower() in supported_formats:
            try:
                # Open image
                with Image.open(image_file) as img:
                    original_size = image_file.stat().st_size

                    # Apply EXIF orientation to actual pixels before stripping metadata
                    img = ImageOps.exif_transpose(img)
                    original_dimensions = img.size

                    # Resize if image is wider than max_width
                    if img.width > max_width:
                        # Calculate new height to maintain aspect ratio
                        aspect_ratio = img.height / img.width
                        new_height = int(max_width * aspect_ratio)
                        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

                    # Convert to RGB if necessary (webp doesn't support all modes)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # Preserve transparency if present
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Create output filename with generic numbering
                    output_file = output_path / f"{converted_count}.webp"

                    # Save as WebP without any metadata
                    # By not passing exif data, we ensure all metadata is stripped
                    img.save(
                        output_file,
                        'WEBP',
                        quality=quality,
                        method=6  # Higher compression effort
                    )

                    new_size = output_file.stat().st_size
                    reduction = ((original_size - new_size) / original_size) * 100

                    print(f"✓ Converted: {image_file.name} → {output_file.name}")
                    print(f"  Dimensions: {original_dimensions[0]}x{original_dimensions[1]} → {img.width}x{img.height}")
                    print(f"  Size: {original_size:,} bytes → {new_size:,} bytes ({reduction:.1f}% reduction)")

                    converted_count += 1

            except Exception as e:
                print(f"✗ Error converting {image_file.name}: {e}")

    print(f"\nConverted {converted_count} images to {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert images to WebP format and remove metadata (including GPS data)"
    )
    parser.add_argument(
        'input_folder',
        help='Path to folder containing images to convert'
    )
    parser.add_argument(
        '-o', '--output',
        default='./cat/pics/',
        help='Output folder (default: ./cat/pics/)'
    )
    parser.add_argument(
        '-q', '--quality',
        type=int,
        default=85,
        choices=range(1, 101),
        metavar='1-100',
        help='WebP quality setting 1-100 (default: 85)'
    )
    parser.add_argument(
        '-w', '--max-width',
        type=int,
        default=1920,
        metavar='PIXELS',
        help='Maximum width in pixels (default: 1920)'
    )

    args = parser.parse_args()

    convert_to_webp(args.input_folder, args.output, args.quality, args.max_width)


if __name__ == '__main__':
    main()
