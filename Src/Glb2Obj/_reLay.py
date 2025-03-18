
"""
Rearrange the LVIS data in the format of cate/ins/ins.glb
Input: lvis_annotations.json ; data/objects/*.glb

"""

import json
import os
import shutil
import glob

def ReLay(data_name='canon' , save_dir="../objaverse_data", info_dir='data/'):  # # data_name = 'lvis', 'canon'

    if data_name == 'canon':
        annotations_path = os.path.join(info_dir, "canon-annotations.json")
    elif data_name == 'lvis':
        annotations_path = os.path.join(save_dir, "lvis-annotations.json")
    object_paths_path = os.path.join(save_dir, "object-paths.json")

    # Read lvis-annotations.json file
    with open(annotations_path, 'r') as file:
        annotations = json.load(file)
    # Get the object-paths.json file
    with open(object_paths_path, 'r') as file:
        object_paths = json.load(file)


    # Create a reLayout directory
    source_dir = save_dir
    target_dir = os.path.join(save_dir, "rearranged_data")
    os.makedirs(target_dir, exist_ok=True)

    # Traverse the categories and file names in annotations
    for category, file_list in annotations.items():
        # Create a catalog for each category
        category_dir = os.path.join(target_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Traverse through the list of files
        for idx, file_name in enumerate(file_list, start=1):
            glb_path = os.path.join(source_dir, object_paths[file_name])

            instance_name = f"{file_name}/{file_name}.glb"
            src_file_path = glb_path
            dest_file_path = os.path.join(category_dir, instance_name)
            if not os.path.exists(os.path.dirname(dest_file_path)):
                os.makedirs(os.path.dirname(dest_file_path))
            
            # Copy the file to a new location and rename it
            if os.path.exists(src_file_path):
                shutil.copy(src_file_path, dest_file_path)
            else:
                print(f"File not found: {src_file_path}")

    print(f"Files rearranged successfully, relayout root is {target_dir}")
    return target_dir



if __name__ == '__main__':
    save_dir="../objaverse_data"
 
    ReLay(data_name='canon', save_dir=save_dir)