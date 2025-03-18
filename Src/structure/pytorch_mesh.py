"""
Used to work with PyTorch3D type mesh

"""
from dataclasses import dataclass
import os
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.ops import sample_points_from_meshes
import glob

from Src.utils.pytorch_utils import add_texture_to_mesh, read_obj_file_and_backup, transformation_mesh, rotate_mesh
from Src.utils.vis import vis_pytorch3d_mesh


# @dataclass
class PytorchMesh:

    @classmethod
    def read_from_obj(cls, file_path: str, device='cuda', backup=True):
        """
        read obj from obj file
        if resave is True, old pt file will be deleted and new pt file will be saved
        if resave is False and pt file exists, pt file will be loaded
        if resave is False and pt file not exists, obj file will be loaded and pt file will be saved
        """
        objs_dirs = os.path.dirname(file_path)  
        objs_name = os.path.basename(file_path).split('.')[0]  # model.obj

        if os.path.exists(objs_dirs+'/'+objs_name+'_verts.pt'):
            verts = torch.load(objs_dirs+'/'+objs_name+'_verts.pt')
            faces = torch.load(objs_dirs+'/'+objs_name+'_faces.pt')
            textures = torch.load(objs_dirs+'/'+objs_name+'_textures.pt')
            if textures is None:
                textures = TexturesVertex(verts_features=torch.ones_like(verts).unsqueeze(0))
            verts = verts.to(device)
            faces = faces.to(device)
            textures = textures.to(device)
            meshes = Meshes(verts=[verts], faces=[faces], textures=textures)
            return cls(meshes)
        else:
            if backup:
                meshes = read_obj_file_and_backup(file_path, device)
            else:
                meshes = load_objs_as_meshes([file_path], device=device)
            return cls(meshes)

    def __init__(self, meshes: Meshes):
        self.meshes = meshes
        # if the mesh dont have texture, add a default texture
        self._add_texture()

    # check the obj is normal to read （the verts number is not 0）
    def check_mesh(self):
        verts = self.meshes.verts_packed()
        # print('[debug]', verts.shape)
        if verts.shape[0] == 0:
            return False
        return True

    # Mesh unitization
    def unitize(self):
        """
        Unitized mesh
        """
        verts = self.meshes.verts_packed()
        # print('[debug]', verts.shape)
        # if verts.shape[0] == 0:
        #     return 0
        verts = verts - verts.mean(dim=0)

        max_distance = torch.max(torch.norm(verts,dim=1))  

        verts = verts / max_distance

        self.meshes = Meshes(verts=[verts], faces=[self.meshes.faces_packed()], textures=self.meshes.textures)
    
    # sample points from mesh
    def to_pts(self, num_points:int):
        """
        sample points from mesh
        return: pts torch.tensor
        """
        pts = sample_points_from_meshes(self.meshes, num_points)
        return pts[0]
    
    # add pose and scale to mesh
    def transform(self, pose:torch.tensor, scale:torch.tensor=None):
        """

        pose: 4*4
        """
        self.meshes = transformation_mesh(self.meshes, pose, scale)

    # rotate the mesh
    def rotate(self, rot:torch.tensor):
        """

        pose: 3*3
        """
        self.meshes = rotate_mesh(self.meshes, rot)

    # if the mesh dont have texture, add a default texture
    def _add_texture(self):
        if self.meshes.textures is None:
            self.meshes = add_texture_to_mesh(self.meshes) 

    # save
    def save(self, save_dir, name):
        """
        save mesh to obj file
        """
        save_path = os.path.join(save_dir, name)
        verts = self.meshes.verts_packed()
        faces = self.meshes.faces_packed()
        textures = self.meshes.textures
        torch.save(verts, save_path+'_verts.pt')
        torch.save(faces, save_path+'_faces.pt')
        torch.save(textures, save_path+'_textures.pt')
    
    def vis(self):
        """
        """
        vis_pytorch3d_mesh(self.meshes)
    
    def centroid(self):
        """
        return the centroid of the mesh
        """
        verts = self.meshes.verts_packed()
        return verts.mean(dim=0)

    def size(self):
        """
        return the size of the mesh
        """
        verts = self.meshes.verts_packed()
        max_xyz = torch.max(verts, dim=0)[0]
        min_xyz = torch.min(verts, dim=0)[0]
        return max_xyz - min_xyz
    def scale(self):
        scale = torch.norm(self.size())
        return scale


if __name__=="__main__":
    obj_path = "results/unit_test/1/model.obj"
    pyObj = PytorchMesh.read_from_obj(obj_path)
    pts = pyObj.to_pts(num_points=1024)
    print(pts.shape)
