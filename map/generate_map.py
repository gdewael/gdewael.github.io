import seaborn as sns
import branca
from PIL import Image
import io
import os
import gpxpy
import pandas as pd
import base64
import folium
import json
import argparse


def load_gpx(file_path):
    with open(file_path) as f:
            gpx = gpxpy.parse(f)

    # Convert to a dataframe one point at a time.
    points = []
    for segment in gpx.tracks[0].segments:
        for p in segment.points:
            points.append({
                'time': p.time,
                'latitude': p.latitude,
                'longitude': p.longitude,
                'elevation': p.elevation,
            })
    df = pd.DataFrame.from_records(points)
    coords = [(i, j) for i, j in zip(df["latitude"], df["longitude"])]
    return coords


def img_to_thumbnail_popup(file_path, tooltip, size = 300):
    buffer = io.BytesIO()
    img = Image.open(file_path)
    img.thumbnail((size, size))  # x, y
    img.save(buffer, format="jpeg")
    encoded = base64.b64encode(buffer.getvalue())

    html = '%s<p><img src="data:image/png;base64,%s">' % (tooltip, encoded.decode('UTF-8'))
    iframe = branca.element.IFrame(html=html, width=325, height = 325)
    return folium.Popup(iframe, max_width=325)

def main():
    parser = argparse.ArgumentParser(
        description="Script to generate a map visualizing my travels on my site.",
    )
    parser.add_argument("--mapping_json_path", type=str, default="./mapping.json")
    parser.add_argument("--img_folder_path", type=str, default="./img")
    parser.add_argument("--gpx_folder_path", type=str, default="./tracks")
    parser.add_argument("--output_file", type=str, default="./treks.html")
    
    args = parser.parse_args()

    m = folium.Map(location = [51.057056, 3.702139], zoom_start = 4, tiles="CartoDB dark_matter")

    t = sns.color_palette().as_hex()
    color_mapping = {
        "multi-day" : t[0],
        "packraft" : t[1],
        "day trip" : t[2],
    }

    with open(args.mapping_json_path) as json_file:
        data = json.load(json_file)

    for ix, l in enumerate(os.listdir(args.gpx_folder_path)):
        if l.endswith(".gpx"):

            coords = load_gpx(os.path.join(args.gpx_folder_path, l))

            tooltip = "%s (%s)" % (data[l][0], data[l][1])
            popup = img_to_thumbnail_popup(os.path.join(args.img_folder_path, data[l][2]), tooltip)
            
            # Outline
            folium.PolyLine(
                coords, weight=8, color = "white",
            ).add_to(m)

            # Colored line
            folium.PolyLine(
                coords, weight=6, color = color_mapping[data[l][-1]],
                tooltip=tooltip, 
                popup=popup,
            ).add_to(m)

    m.save(args.output_file)

if __name__ == "__main__":
    main()
