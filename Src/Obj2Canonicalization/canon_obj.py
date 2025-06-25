"""
    Transform the mesh to a canonical field
"""


import json
import numpy as np
import os
import glob
import torch
import trimesh
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def obtain_canonLabel():
    """Load canonical pose labels (assuming labels are 3x3 rotation matrices)"""
    canon_lab_dict = {}
    with open('data/CanonicalObjaverseDataset.json', 'r') as file:
        allData = json.load(file)
    for category, uid_pose_list in allData.items():
        for uid_pose in uid_pose_list:  # Each entry is [uid, list of 3x3 rotation matrices]
            uid = uid_pose[0]
            pose_matrix = torch.tensor(uid_pose[1], dtype=torch.float32).view(3, 3)
            canon_lab_dict[uid] = pose_matrix
    return canon_lab_dict

def apply_3x3_rotation(mesh: trimesh.Trimesh, rotation_matrix: torch.Tensor):
    """Apply 3x3 rotation matrix to mesh (first convert to 4x4 homogeneous matrix)"""
    homogenous_matrix = torch.zeros((4, 4), dtype=torch.float32, device=rotation_matrix.device)
    homogenous_matrix[:3, :3] = rotation_matrix
    homogenous_matrix[3, 3] = 1.0
    transform = homogenous_matrix.numpy()
    mesh.apply_transform(transform)
    return mesh

def process_single_mesh(obj_path: str, canon_lab_dict: dict):
    """Function to process a single mesh (for multithreaded use)"""
    uid = Path(obj_path).stem
    if uid not in canon_lab_dict:
        print(f"Warning: Canonical pose label not found for UID {uid}")
        return
    
    can_pose = canon_lab_dict[uid]
    try:
        mesh = trimesh.load(obj_path, process=False)
        vertices = mesh.vertices
        center = vertices.mean(axis=0)
        scale = np.max(np.linalg.norm(vertices - center, axis=1))
        if scale > 0:
            mesh.vertices = (vertices - center) / scale
        transformed_mesh = apply_3x3_rotation(mesh, can_pose)
        Path(obj_path).parent.mkdir(parents=True, exist_ok=True)
        transformed_mesh.export(obj_path, file_type='obj')
    except Exception as e:
        print(f"Processing failed for {obj_path}: {e}")

def canon_meshs(save_dir: str):
    """Batch process meshes (multithreaded version)"""
    obj_files = glob.glob(os.path.join(save_dir, 'rearranged_data', '**/*.obj'), recursive=True)
    canon_lab_dict = obtain_canonLabel()
    
    # Determine number of threads (recommended 2-4x CPU cores, adjust based on system)
    num_threads = min(os.cpu_count() * 2, 16)  # Example: max 16 threads
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Wrap multithreaded processing with tqdm for progress bar
        list(tqdm(executor.map(lambda p: process_single_mesh(p, canon_lab_dict), obj_files), 
                 total=len(obj_files), desc="Processing Progress"))

if __name__ == "__main__":
    save_dir = "results"
    canon_meshs(save_dir)

# import json
# import numpy as np
# import os
# import glob 
# import torch

# from Src.structure.pytorch_mesh import PytorchMesh
# from Src.utils.pytorch_utils import rewrite_objFile


# def obtain_canonLabel():
#     canon_lab_dict = {}
#     with open('data/CanonicalObjaverseDataset.json', 'r') as file:
#         allData = json.load(file)
#     # canon-annotations
#     for category, uid_pose in allData.items():
#         for uid, pose in uid_pose:
#            canon_lab_dict[uid] = torch.tensor(pose)
#     return canon_lab_dict


# def canon_mesh(obj_path:str, can_pose:torch.tensor, save_path:str=None):
#     """
#     canonical the mesh and rewrite the obj file
#     """
#     # unit
#     pyObj = PytorchMesh.read_from_obj(obj_path, backup=False)
#     # check mesh is normal
#     if pyObj.check_mesh() is False:
#         # print(obj_path)
#         # asdf
#         print('mesh is not normal')
#         return None
#     pyObj.unitize()

#     # canonical_rot
#     device = pyObj.meshes.device
#     pyObj.rotate(can_pose.to(device))

#     # write2obj
#     rewrite_objFile(obj_path, pyObj.meshes.verts_packed(), save_path=save_path)
#     return pyObj


# def canon_meshs(save_dir):
#     # Get the path to all obj
#     obj_files = glob.glob(os.path.join(save_dir, 'rearranged_data', '**/*.obj'), recursive=True)
#     # Get all of Canon Labs
#     canon_lab_dict = obtain_canonLabel()
#     # Traversal canonicalization
#     for obj_path in obj_files:
#         save_path = obj_path
#         uid = obj_path.split('/')[-1].split('.obj')[0]
#         can_pose = canon_lab_dict[uid]
#         canon_mesh(obj_path, can_pose, save_path)





# if __name__=="__main__":
#     save_dir="../objaverse_data"
#     canon_meshs(save_dir)



