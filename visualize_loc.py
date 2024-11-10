# we want to map all lat and longs as dots on a map of SF

import json

# load from tmp_datasets/bdd100k/sf_images.json

with open('tmp_datasets/bdd100k/sf_images.json') as f:
    sf_images = json.load(f)
    
# key is image and value is lat and long as tuple

# we dont care about the image name, just the lat and long
# now plot onto a map
import folium

# create a map of SF
m = folium.Map(location=[37.7749, -122.4194], zoom_start=12)

for k, v in sf_images.items():
    folium.Marker(v, popup=k).add_to(m)
    
m.save('sf_images.html')
print('Map saved to sf_images.html')
print('Open the file in a browser to view the map')
