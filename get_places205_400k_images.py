import json
import os
import shutil

def check_paths(local,database,file):
    # Check if the local path exists
    if not os.path.isdir(local):
        raise FileNotFoundError(f"Local path {local} does not exist")
    # Check if the database path exists
    if not os.path.isdir(database):
        raise FileNotFoundError(f"Database path {database} does not exist")
    # Check if the file exists
    if not os.path.isfile(file):
        raise FileNotFoundError(f"File {file} does not exist")

def get_places205_400k_images(local_path,database_path,path):
    
    check_paths(local_path, database_path, path)
    
    with open(path) as f:
        dataset = json.load(f)

    count = 0
    count_already = 0
    print("Total images in:", path, ":", len(dataset['data']))
    
    for instance in dataset['data']:
        image_path = instance['image']
        # Check if the image exists in the database
        if not os.path.isfile(os.path.join(database_path, image_path)):
            print(f"Image {image_path} does not exist in the database")
            count += 1
            continue

        # Check if the image exists in the local path
        if os.path.isfile(os.path.join(local_path, image_path)):
            count_already += 1
            continue

        # Get the category path
        category_path = os.path.dirname(image_path)
        full_category_path = os.path.join(local_path, category_path)
        
        # Create the folders if they don't exist
        os.makedirs(full_category_path, exist_ok=True)

        # Copy the image
        shutil.copyfile(os.path.join(database_path, image_path), os.path.join(local_path, image_path))
        print("|", end="")
    print()
    print(f"Total images not found: {count}")
    print(f"Total images already in the local path: {count_already}")
    print("Total images copied:", len(dataset['data']) - count - count_already)
    print("That's all folks!")

if __name__ == '__main__':
    # Move to home
    os.chdir('/home/asantos')
    local_path = '/data/upftfg03/asantos/PlacesAudio_400k_distro/images256'
    database_path = '/data/upftfg03/asantos/OpenDataLab___Places205/raw/data/vision/torralba/deeplearning/images256'
    path='/data/upftfg03/asantos/PlacesAudio_400k_distro/metadata/val.json'
    get_places205_400k_images(local_path,database_path,path)

    