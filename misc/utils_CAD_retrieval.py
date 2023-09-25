import copy
import os

import numpy as np
import open3d as o3d
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d

from misc.utils import SEMANTIC_IDX2NAME, COLOR_DETECTRON2
from misc.line_mesh import LineMesh

def normalize_mesh(verts_init, faces, device):
    mesh = Meshes(
        verts=[verts_init],
        faces=[faces],
    )
    bbox = mesh.get_bounding_boxes().squeeze(dim=0)
    bbox = bbox.cpu().detach().numpy()

    center = torch.tensor(bbox.mean(1)).float().to(device)
    vector_x = np.array([bbox[0, 1] - bbox[0, 0], 0, 0])
    vector_y = np.array([0, bbox[1, 1] - bbox[1, 0], 0])
    vector_z = np.array([0, 0, bbox[2, 1] - bbox[2, 0]])

    coeff_x = np.linalg.norm(vector_x)
    coeff_y = np.linalg.norm(vector_y)
    coeff_z = np.linalg.norm(vector_z)

    mesh = mesh.offset_verts(-center)
    transform_func = Transform3d().scale(x=(1 / coeff_x), y=(1 / coeff_y), z=1 / coeff_z).to(device)
    tverts = transform_func.transform_points(mesh.verts_list()[0]).unsqueeze(dim=0)

    mesh = Meshes(
        verts=[tverts.squeeze(dim=0)],
        faces=[faces]
    )

    return mesh, tverts


def cut_meshes(mesh_o3d, indices_list):
    mesh_o3d_obj = copy.deepcopy(mesh_o3d)
    mesh_o3d_obj = mesh_o3d_obj.select_by_index(indices_list)
    mesh_o3d.remove_vertices_by_index(indices_list)

    face_list_bg = np.asarray(mesh_o3d.triangles)
    face_list_obj = np.asarray(mesh_o3d_obj.triangles)

    mesh_bg = Meshes(
        verts=[torch.tensor(np.asarray(mesh_o3d.vertices)).float()],
        faces=[torch.tensor(np.asarray(face_list_bg))]
    )

    mesh_obj = Meshes(
        verts=[torch.tensor(np.asarray(mesh_o3d_obj.vertices)).float()],
        faces=[torch.tensor(np.asarray(face_list_obj))]
    )
    return mesh_bg, mesh_obj


def load_textured_cad_model_prepro(model_path, cad_transform_base, cls_name, device='cpu'):
    """Load CAD models from ShapeNet_preprocessed (using run_shapenet_prepro.sh)."""

    try:
        sem_id = list(SEMANTIC_IDX2NAME.keys())[
            list(SEMANTIC_IDX2NAME.values()).index(cls_name)]
    except:
        sem_id = 0
    mesh_color = COLOR_DETECTRON2[sem_id]

    verts, faces_, _ = load_obj(model_path, load_textures=False, device=device)
    faces = faces_.verts_idx

    _, tverts = normalize_mesh(verts, faces, device)

    tverts_final = cad_transform_base.transform_points(tverts)

    if device == 'cpu':
        tverts_final_ary = tverts_final.squeeze(dim=0).detach().numpy()
        faces_ary = faces.detach().numpy().astype(np.int32)

    else:
        tverts_final_ary = tverts_final.squeeze(dim=0).cpu().detach().numpy()
        faces_ary = faces.cpu().detach().numpy().astype(np.int32)

    cad_model_o3d = o3d.geometry.TriangleMesh()
    cad_model_o3d.vertices = o3d.utility.Vector3dVector(tverts_final_ary)
    cad_model_o3d.triangles = o3d.utility.Vector3iVector(faces_ary)
    vertex_n = np.array(cad_model_o3d.vertices).shape[0]
    vertex_colors = np.ones((vertex_n, 3)) * mesh_color
    cad_model_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    return cad_model_o3d


def load_textured_cad_model(model_path, cad_transform_base, cls_name, device='cpu'):
    """Load CAD models from raw ShapeNetCore.v2 and scale-normalize + center model."""

    try:
        sem_id = list(SEMANTIC_IDX2NAME.keys())[
            list(SEMANTIC_IDX2NAME.values()).index(cls_name)]
    except:
        sem_id = 0
    mesh_color = COLOR_DETECTRON2[sem_id]

    verts, faces_, _ = load_obj(model_path, load_textures=False, device=device)
    faces = faces_.verts_idx

    _, tverts = normalize_mesh(verts, faces, device)

    tverts_final = cad_transform_base.transform_points(tverts)

    if device == 'cpu':
        tverts_final_ary = tverts_final.squeeze(dim=0).detach().numpy()
        faces_ary = faces.detach().numpy().astype(np.int32)

    else:
        tverts_final_ary = tverts_final.squeeze(dim=0).cpu().detach().numpy()
        faces_ary = faces.cpu().detach().numpy().astype(np.int32)

    cad_model_o3d = o3d.geometry.TriangleMesh()
    cad_model_o3d.vertices = o3d.utility.Vector3dVector(tverts_final_ary)
    cad_model_o3d.triangles = o3d.utility.Vector3iVector(faces_ary)
    vertex_n = np.array(cad_model_o3d.vertices).shape[0]
    vertex_colors = np.ones((vertex_n, 3)) * mesh_color
    cad_model_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    return cad_model_o3d


def drawOpen3dCylLines(bbListIn,col=None):

    lines = [[0, 1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
             [0,4],[1,5],[2,6],[3,7]]

    line_sets = []

    for bb in bbListIn:
        points = bb
        if col is None:
            col = [0,0,1]
        colors = [col for i in range(len(lines))]

        line_mesh1 = LineMesh(points, lines, colors, radius=0.02)
        line_mesh1_geoms = line_mesh1.cylinder_segments
        line_sets = line_mesh1_geoms[0]
        for l in line_mesh1_geoms[1:]:
            line_sets = line_sets + l

    return line_sets