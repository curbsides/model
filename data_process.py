# script to process data
# tmp_datasets/bdd100k/images/100k/train contains 7000 images
# load the image names into a dictionary

import os

def load_image_names():
    # path to the image folder
    image_folder = 'tmp_datasets/bdd100k/images/100k/train'
    # list all the images
    image_names = os.listdir(image_folder)
    
    # remove file extension
    image_names = [image.split('.')[0] for image in image_names]

    # return set of image names
    return set(image_names)

# load info files into a dictionary
def load_info_files():
    # path to the info folder
    info_folder = 'tmp_datasets/bdd100k/info/'
    # list all the info files
    info_files = os.listdir(info_folder)
    
    # remove file extension
    info_files = [file.split('.')[0] for file in info_files]
    
    return set(info_files)

# test on one random image
if __name__ == '__main__':
    import json
    # find the intersection of the two sets
    image_names = load_image_names()
    info_files = load_info_files()
    intsect = image_names.intersection(info_files)
    
    print('Number of images: ', len(image_names))
    print('Number of info files: ', len(info_files))
    print('Number of intersection: ', len(intsect))
    
    # load tmp_datasets/bdd100k/labels/det_train.json
    with open ('tmp_datasets/bdd100k/labels/det_train.json') as f:
        det = json.load(f)
        
    # make a new dict with key as name and value as scene only if scene == 'city street'
    scene_dict = {}
    for d in det:
        if  d['attributes']['timeofday'] == 'daytime':
            # split name and remove extension
            sm = d['name'].split('.')[0]
            scene_dict[sm] = d['attributes']['scene']
    # intersection with the intsect
    scene_keys = set(scene_dict.keys())
    intsect_scene = scene_keys.intersection(intsect)
    print('Number of city street images after restricting attribute: ', len(intsect_scene))
    
    # restrict to just certain cities i.e. sf
    # upper left = [37.80085781617528, -122.50777363570397]
    # bottom right = [37.71237494683714, -122.39306117661563]
    # check if location is within the range
    upper_left = (37.80085781617528, -122.50777363570397)
    bottom_right = (37.71237494683714, -122.39306117661563)
    final_set = set()
    save_dict = {}
    for name in intsect_scene:
        json_info = os.path.join('tmp_datasets/bdd100k/info/', name+'.json')
        with open(json_info) as f:
            data = json.load(f)
        if not('locations' in data and len(data['locations']) > 0):
            continue
        if not('latitude' in data['locations'][0] and 'longitude' in data['locations'][0]):
            continue
        location = (data['locations'][0]['latitude'], data['locations'][0]['longitude'])
        
        if location[0] < upper_left[0] and location[0] > bottom_right[0] and location[1] > upper_left[1] and location[1] < bottom_right[1]:
            final_set.add(name)
            save_dict[name] = location
    print('Number of images after restricting to SF: ', len(final_set))
    
    # save the dictionary
    with open('tmp_datasets/bdd100k/sf_images.json', 'w') as f:
        json.dump(save_dict, f)
    
    json_path = 'tmp_datasets/bdd100k/info/'
    name = list(final_set)[0]
    # pick one random intersection
    json_info = os.path.join(json_path, name+'.json')
    # load the json file
    with open(json_info) as f:
        data = json.load(f)
    
    # show the image
    from PIL import Image
    image_path = 'tmp_datasets/bdd100k/images/100k/train/'+name+'.jpg'
    print('image path', image_path)
    print('scene: ', scene_dict[name])
    print('name: ', name)
    print('location: ', data['locations'][0])
    img = Image.open(image_path)
    img.show()

    # now move all SF images to a new folder for training
    path = 'tmp_datasets/bdd100k/sf_images'
    if not os.path.exists(path):
        os.mkdir(path)
    for name in final_set:
        # copy image from tmp_datasets/bdd100k/images/100k/train to tmp_datasets/bdd100k/sf_images
        os.system('cp tmp_datasets/bdd100k/images/100k/train/'+name+'.jpg '+path +'/'+name+'.jpg')
        