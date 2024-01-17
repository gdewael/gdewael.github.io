---
title: Mapping travels with Folium
description: This post describes the process of making a hikes of treks using Folium
date: 2024-01-18
tags:
  - personal
---

There's a map at my grandparents' home containing sewing pins that indicate the places they have traveled.
As a kid, I used to gawk at this map and wonder how these places looked like.
As an adult, I have come to appreciate this map as a testament to a life well-spent traveling together.

<p style="text-align: center;">
  {% image "./map.png", "A map", "700" %}
  <br>
  <i>The map at my grandparents' house. Every sewing pin indicates a visited place.</i>
</p>

As an homage to their map (and to satisfy my own love for staring at maps and tracking data), I have decided to similarly start tracking travels.
Instead of using sewing pins, I will be hosting it on this site.
Currently, the map contains only multi-day trails.
I may expand this with dots for places visited using more-conventional means of travel, but for the time being, I'm content with how it looks.
The result is:

<p style="text-align: center;">
  <iframe src="../../treks/index.html" width="700" height="450"></iframe>
  <br>
  <i>A map containing my multi-day trails</i>
</p>


As the map itself is very much a work in progress, the remainder of this page will outline how I created the map, and how I integrated it with this site.

## Overview and file prerequisites

The setup consists of a python file that (1) reads *.gpx* and *.jpg* files, and (2) renders it on a Leaflet.js map using [folium](https://python-visualization.github.io/folium/latest/).
To get *.gpx* files for hikes, I use the [Outdooractive](https://www.outdooractive.com/en/) platform, which is also my mobile app during hikes.
With my current workflow, the folder structure looks as follows:

```bash
map/
├── generate_map.py
├── mapping.json
├── requirements.txt
├── img
│   └── *.jpg
└── tracks
    └── *.gpx
```

In a *mapping.json* file, I can record some metadata for every track:

```bash
> cat mapping.json
{
    "camino_primitivo.gpx": ["12-day Camino Primitivo", "August 2015", "camino.JPG", "multi-day"],
    ...
    "tarn.gpx": ["2-day Packrafting of the Tarn", "July 2023", "tarn.jpg", "packraft"]
}
```
Currently, for every *.gpx* file, I'm tracking a short description of the trail, a description of the date, an image file, and what kind of trail it corresponds to.
All of these attributes are used in the tooltips of the trails on the map.

In order to add new tracks, I add a *.gpx* track, an image, and a new entry in the *mapping.json* file.
In terms of ease-of-use, it's not quite up-to-par with sticking a sewing pin on a paper map, but let's hope it stands the test of time.

## Mapping with Folium

The script is kept deliberately simple.
It creates a Folium map:

<details><summary>Code for folium map</summary>

```python
m = folium.Map(location = [51.057056, 3.702139], zoom_start = 4, tiles="CartoDB dark_matter")

Fullscreen(
		position="topright",
		title="Expand me",
		title_cancel="Exit me",
		force_separate_button=True,
).add_to(m)
```

</details>

After which it loops over the entries in the metadata *mapping.json* files:

<details><summary>Code for adding tracks to map</summary>

```python
colors = sns.color_palette().as_hex()
color_mapping = {
		"multi-day" : colors[0],
		"packraft" : colors[1],
		"day trip" : colors[2],
}

with open(mapping_json_path) as json_file:
		data = json.load(json_file)


for ix, l in enumerate(os.listdir(gpx_folder_path)):
		if l.endswith(".gpx"):
				name, date, pic, color = data[l]

				coords = load_gpx(os.path.join(gpx_folder_path, l))

				tooltip = "%s (%s)" % (name, date)
				popup = img_to_thumbnail_popup(os.path.join(img_folder_path, pic), tooltip)

				# Outline
				folium.PolyLine(
						coords, weight=8, color = "white",
				).add_to(m)

				# Colored line
				folium.PolyLine(
						coords, weight=6, color = color_mapping[color],
						tooltip=tooltip,
						popup=popup,
				).add_to(m)

m.save(output_file)
```

</details>

This piece of code makes use of a function *load_gpx*, that loads *.gpx* files to pandas DataFrames:

<details><summary>Code for reading gpx files</summary>

```python
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
```

</details>

In addition, pictures are processed using a *img_to_thumbnail_popup* function.
Of note here is that the images in the popups are programmed to be Base64 encoded and served inline.
This is the quickest way I've gotten it set-up, but I realize that this will need to be changed in the future for scalability reasons.

<details><summary>Code for creating thumbnails</summary>

```python
def img_to_thumbnail_popup(file_path, tooltip, size = 300):
    buffer = io.BytesIO()
    img = Image.open(file_path)
    img.thumbnail((size, size))  # x, y
    img.save(buffer, format="jpeg")
    encoded = base64.b64encode(buffer.getvalue())

    html = '%s<p><img src="data:image/png;base64,%s">' % (tooltip, encoded.decode('UTF-8'))
    iframe = branca.element.IFrame(html=html, width=325, height = 325)
    return folium.Popup(iframe, max_width=325)
```

</details>

The full script can be found [here](https://github.com/gdewael/gdewael.github.io/blob/main/map/generate_map.py).

## Integration with GitHub Pages static site

The script writes the map to a *.html* file, which can be readily used in my static site, using iframes, for example.
I have the Eleventy build of this site hooked up to GitHub Actions, so why not do the same for this map?
This way, I don't have to run the python script manually every time I add a new trail.
To do this, I add the following "steps" in [my GitHub Actions workflow file](https://github.com/gdewael/gdewael.github.io/blob/main/.github/workflows/build-and-deploy.yml):

```yml
- name: setup python
	uses: actions/setup-python@v4
	with:
		python-version: '3.12' # install the python version needed

- name: install python packages
  run: |
		python -m pip install --upgrade pip
		pip install -r ./map/requirements.txt

- name: execute py script # run main.py
	run: python ./map/generate_map.py --mapping_json_path map/mapping.json --img_folder_path map/img/ --gpx_folder_path map/tracks/ --output_file content/treks.html

```

To add a new trail, I just push a new entry in the *metadata.json* file, along with adding an image and gpx file, to GitHub.
Upon doing so, the workflow triggers, automatically updating the map and the site along with it.

I will be hosting a stand-alone page of the map [here](../../map).
