"""
GLB to OBJ file
"""

import os
import cv2
import json
import torch
import trimesh
import glob
import time
from tqdm import tqdm
import multiprocessing
from pathlib import Path

from Src.utils.parallel_cpu import *
from Src.Glb2Obj._reLay import ReLay


class Mesh:
    def __init__(self, mesh_path, target_scale=1.0, mesh_dy=0.0,
                 remove_mesh_part_names=None, remove_unsupported_buffers=None, intermediate_dir=None):
        # from https://github.com/threedle/text2mesh
        self.material_cvt, self.material_num, org_mesh_path, is_convert = None, 1, mesh_path, False
        if not mesh_path.endswith(".obj") and not mesh_path.endswith(".off"):
            if mesh_path.endswith(".gltf"):
                mesh_path = self.preprocess_gltf(mesh_path, remove_mesh_part_names, remove_unsupported_buffers)
            mesh_temp = trimesh.load(mesh_path, force='mesh', process=True, maintain_order=True)

            mesh_path = os.path.splitext(mesh_path)[0] + ".obj"
            mesh_temp.export(mesh_path)
            merge_texture_path = os.path.join(os.path.dirname(mesh_path), "material_0.png")
            if os.path.exists(merge_texture_path):
                self.material_cvt = cv2.imread(merge_texture_path)
                self.material_num = self.material_cvt.shape[1] // self.material_cvt.shape[0]
            # logger.info("Converting current mesh model to obj file with {} material~".format(self.material_num))
            print("Converting current mesh model to obj file with {} material~".format(self.material_num))


    def preprocess_gltf(self, mesh_path, remove_mesh_part_names, remove_unsupported_buffers):
        with open(mesh_path, "r") as fr:
            gltf_json = json.load(fr)
            if remove_mesh_part_names is not None:
                temp_primitives = []
                for primitive in gltf_json["meshes"][0]["primitives"]:
                    if_append, material_id = True, primitive["material"]
                    material_name = gltf_json["materials"][material_id]["name"]
                    for remove_mesh_part_name in remove_mesh_part_names:
                        if material_name.find(remove_mesh_part_name) >= 0:
                            if_append = False
                            break
                    if if_append:
                        temp_primitives.append(primitive)
                gltf_json["meshes"][0]["primitives"] = temp_primitives
                print("Deleting mesh with materials named '{}' from gltf model ~".format(remove_mesh_part_names))

            if remove_unsupported_buffers is not None:
                temp_buffers = []
                for buffer in gltf_json["buffers"]:
                    if_append = True
                    for unsupported_buffer in remove_unsupported_buffers:
                        if buffer["uri"].find(unsupported_buffer) >= 0:
                            if_append = False
                            break
                    if if_append:
                        temp_buffers.append(buffer)
                gltf_json["buffers"] = temp_buffers
                print("Deleting unspported buffers within uri {} from gltf model ~".format(remove_unsupported_buffers))
            updated_mesh_path = os.path.splitext(mesh_path)[0] + "_removed.gltf"
            with open(updated_mesh_path, "w") as fw:
                json.dump(gltf_json, fw, indent=4)
        return updated_mesh_path

    def normalize_mesh(self, target_scale=1.0, mesh_dy=0.0):
        print('in mesh normalization, the target scale is ', target_scale)
        verts = self.vertices
        center = verts.mean(dim=0)
        print(center)
        verts = verts - center        
        
        scale = torch.max(torch.norm(verts, p=2, dim=1))   
        print('scale is ', scale)
        print('target_scale is ', target_scale)
        verts = verts *  target_scale
        
   
        
        verts[:, 1] += mesh_dy   
        print('mesh_dy is ', mesh_dy)
        self.vertices = verts


# Convert to obj
def Glb2Obj(glb_files,n, save_dir):
    for i, glb_path in enumerate(tqdm(glb_files,desc=f'pid:{str(os.getpid())[-5:]}:',position=n)):
        try:
            mesh=Mesh(glb_path)
            del mesh
        except:
            log_path = os.path.join(save_dir, 'log', 'log.txt')
            os.makedirs(os.path.join(save_dir, 'log'), exist_ok=True)
            with open(log_path, 'a') as f:
                f.write(f'err glb2Obj path: {glb_path} \n')


# relay and glb2Obj
def Relay2Obj(save_dir):
    ### reLay
    reLayRoot = ReLay(data_name='canon', save_dir=save_dir)
    ### glb2Obj
    glb_files = glob.glob(os.path.join(reLayRoot, '**/*.glb'), recursive=True)

    '''# debug
    for glb_path in glb_files:
        mesh=Mesh(glb_path)
        del mesh
        print('glb_path:', glb_path)
        asdf'''


    pool = multiprocessing.Pool(multiprocessing.cpu_count()//2) # Create a process pool with the parameter set to CPU cores/2
    glb_files_list=divide_list(glb_files, multiprocessing.cpu_count()//2)
    # Start multiple processes
    for i in range(multiprocessing.cpu_count()//2):
        # Assign tasks to process pools
        pool.apply_async(Glb2Obj, args=(glb_files_list[i], i, save_dir))
    # Close the process pool
    pool.close()
    pool.join()


if __name__=="__main__":

    save_dir="../objaverse_data"
    Relay2Obj(save_dir)
    
    


    
    