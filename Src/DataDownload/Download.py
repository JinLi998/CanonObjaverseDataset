import Src.DataDownload._obja as objaverse 
import os
import json
import gzip
# from huggingface_hub import hf_hub_download
os.environ['CURL_CA_BUNDLE'] = ''

def get_glbs(data_name='canon'):  # data_name = 'lvis', 'canon', '1.0'
    lvis_annotations = objaverse.load_lvis_annotations()
    uids = []
    
    if data_name == 'lvis':
        categories = list(lvis_annotations.keys())
        for category in categories:
            uids += lvis_annotations[category]
    elif data_name == 'canon':
        # Read the data/CanonicalObjaverseDataset.json file
        with open('data/CanonicalObjaverseDataset.json', 'r') as file:
            allData = json.load(file)
        categories = list(allData.keys())
        # canon-annotations
        canon_annos = {}
        for category, uid_pose in allData.items():
            canon_annos[category] = [u_pose[0] for u_pose in uid_pose]
            uids.extend(canon_annos[category])
        # canon-annotations.json 
        with open('data/canon-annotations.json', 'w') as file:
            json.dump(canon_annos, file)
    elif data_name == 'test':
        # Read the data/CanonicalObjaverseDataset.json file
        with open('data/CanonicalObjaverseDataset.json', 'r') as file:
            allData = json.load(file)
        categories = list(allData.keys())
        # canon-annotations
        canon_annos = {}
        for category, uid_pose in allData.items():
            if category != 'chair':
                continue
            canon_annos[category] = [u_pose[0] for u_pose in uid_pose]
            uids.extend(canon_annos[category])
        # canon-annotations.json 
        with open('data/canon-annotations.json', 'w') as file:
            json.dump(canon_annos, file)
    elif data_name == '1.0':
        object_paths = objaverse._load_object_paths()
        uids = list(object_paths.keys())

    print('Total UIDs:', len(uids))

    
   # Download the data
    processes = 60 #mp.cpu_count()
    glbs = objaverse.load_objects(uids, processes)
    return glbs


def DownloadGLBs(data_name='canon', save_dir="../objaverse_data"):
    objaverse._VERSIONED_PATH = save_dir
    glbs = get_glbs(data_name)
    # 解压
    json_path = os.path.join(save_dir,'object-paths.json.gz')
    with gzip.open(json_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    output_file_path = os.path.join(save_dir,'object-paths.json')
    with open(output_file_path, 'w', encoding='utf-8') as out_f:
        json.dump(data, out_f, ensure_ascii=False, indent=4)
    return glbs

if __name__ == '__main__':
    # Download the Canonical Objaverse Dataset
    DownloadGLBs(data_name='canon', save_dir="../objaverse_data")