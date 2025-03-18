"""
    Transform the mesh to a canonical field
"""

import json
import numpy as np
import os
import glob 
import torch

from Src.structure.pytorch_mesh import PytorchMesh
from Src.utils.pytorch_utils import rewrite_objFile


def obtain_canonLabel():
    canon_lab_dict = {}
    with open('data/CanonicalObjaverseDataset.json', 'r') as file:
        allData = json.load(file)
    # canon-annotations
    for category, uid_pose in allData.items():
        for uid, pose in uid_pose:
           canon_lab_dict[uid] = torch.tensor(pose)
    return canon_lab_dict


def canon_mesh(obj_path:str, can_pose:torch.tensor, save_path:str=None):
    """
    canonical the mesh and rewrite the obj file
    """
    # unit
    pyObj = PytorchMesh.read_from_obj(obj_path, backup=False)
    # check mesh is normal
    if pyObj.check_mesh() is False:
        # print(obj_path)
        # asdf
        print('mesh is not normal')
        return None
    pyObj.unitize()

    # canonical_rot
    device = pyObj.meshes.device
    pyObj.rotate(can_pose.to(device))

    # write2obj
    rewrite_objFile(obj_path, pyObj.meshes.verts_packed(), save_path=save_path)
    return pyObj


def canon_meshs(save_dir):
    # Get the path to all obj
    obj_files = glob.glob(os.path.join(save_dir, 'rearranged_data', '**/*.obj'), recursive=True)
    # Get all of Canon Labs
    canon_lab_dict = obtain_canonLabel()
    # Traversal canonicalization
    for obj_path in obj_files:
        save_path = obj_path
        uid = obj_path.split('/')[-1].split('.obj')[0]
        can_pose = canon_lab_dict[uid]
        canon_mesh(obj_path, can_pose, save_path)





if __name__=="__main__":
    save_dir="../objaverse_data"
    canon_meshs(save_dir)



