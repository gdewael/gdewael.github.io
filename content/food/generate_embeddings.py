#!/usr/bin/env python3
"""
Generate text embeddings for food gallery descriptions.
Uses HuggingFace Inference API with a text embedding model.
"""

import os
import json
import argparse
import time
from datetime import datetime, timezone

try:
    from huggingface_hub import InferenceClient
except ImportError:
    print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
    exit(1)


def generate_embeddings(food_dir, mapping_file, output_file, hf_token, incremental=False):
    """Generate text embeddings for all food descriptions."""

    print("=" * 60)
    print("Starting text embedding generation...")
    print(f"Food directory: {food_dir}")
    print(f"Mapping file: {mapping_file}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    print()

    # Initialize HuggingFace client
    client = InferenceClient(token=hf_token)

    # Use a reliable text embedding model
    model = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Using model: {model}")
    print()

    # Load mapping (date -> description)
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    # Load existing embeddings if incremental update
    existing_embeddings = {}
    if incremental and os.path.exists(output_file):
        print("Loading existing embeddings for incremental update...")
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            for emb in existing_data.get("embeddings", []):
                existing_embeddings[emb["id"]] = emb
        print(f"Loaded {len(existing_embeddings)} existing embeddings")
        print()

    # Find all .webp files and sort by number
    webp_files = sorted(
        [f for f in os.listdir(food_dir) if f.endswith('.webp')],
        key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0
    )

    print(f"Found {len(webp_files)} images")
    print()

    embeddings = []
    processed = 0
    skipped = 0
    errors = 0

    # Process each image's description
    descriptions_list = list(mapping.items())

    for webp_file in webp_files:
        # Extract ID from filename (e.g., "42.webp" -> 42)
        file_id = int(webp_file.split('.')[0])

        # Skip if already processed in incremental mode
        if incremental and file_id in existing_embeddings:
            embeddings.append(existing_embeddings[file_id])
            skipped += 1
            if skipped % 20 == 0:
                print(f"Skipped {skipped} existing embeddings...")
            continue

        # Get the corresponding description
        if file_id - 1 < len(descriptions_list):
            date, description = descriptions_list[file_id - 1]
        else:
            print(f"Warning: No description found for {webp_file}, skipping...")
            errors += 1
            continue

        print(f"Processing {file_id}/{len(webp_files)}: {webp_file} ({date})")
        print(f"  Description: {description[:60]}...")

        try:
            # Generate text embedding using HuggingFace InferenceClient
            print(f"  Generating embedding...")
            embedding = client.feature_extraction(
                text=description,
                model=model
            )

            # Convert numpy array to list if needed
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()

            # If embedding is 2D (batch), take first element
            if isinstance(embedding, list) and isinstance(embedding[0], list):
                embedding = embedding[0]

            # Store embedding data
            embedding_entry = {
                "id": file_id,
                "image_path": f"./{webp_file}",
                "date": date,
                "description": description,
                "embedding": embedding
            }

            embeddings.append(embedding_entry)
            processed += 1

            print(f"  ✓ Success (embedding dim: {len(embedding)})")
            print()

            # Small delay to avoid rate limiting
            if processed % 10 == 0:
                time.sleep(1)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            print(f"  Skipping {webp_file}")
            print()
            errors += 1
            continue

    # Sort by ID to maintain order
    embeddings.sort(key=lambda x: x["id"])

    # Create output structure
    output_data = {
        "version": "1.0",
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embeddings": embeddings
    }

    # Save to file
    print(f"Saving {len(embeddings)} embeddings to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Calculate file size
    file_size = os.path.getsize(output_file) / 1024  # KB

    print()
    print("=" * 60)
    print("✓ Embedding generation complete!")
    print(f"  Total processed: {processed}")
    print(f"  Total skipped: {skipped}")
    print(f"  Total errors: {errors}")
    print(f"  Total embeddings: {len(embeddings)}")
    print(f"  Output file: {output_file}")
    print(f"  File size: {file_size:.1f} KB")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate text embeddings for food gallery descriptions."
    )
    parser.add_argument(
        "--food_dir",
        type=str,
        default="content/food",
        help="Directory containing food images"
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        default="content/food/mapping.json",
        help="JSON file mapping dates to descriptions"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="content/food/embeddings.json",
        help="Output file for embeddings"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="HuggingFace API token (get one at https://huggingface.co/settings/tokens)"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only process new/changed images (skip existing embeddings)"
    )

    args = parser.parse_args()

    # Generate embeddings
    generate_embeddings(
        food_dir=args.food_dir,
        mapping_file=args.mapping_file,
        output_file=args.output_file,
        hf_token=args.hf_token,
        incremental=args.incremental
    )


if __name__ == "__main__":
    main()
