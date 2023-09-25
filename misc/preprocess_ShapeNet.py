import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Data preprocessing for ShapeNetCore.v2")
parser.add_argument("--device", type=str, default="", help="device")

os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().device

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import numpy as np
from pytorch3d.io import IO
from misc.utils import shapenet_category_dict
import trimesh
import open3d as o3d
from misc.utils_CAD_retrieval import normalize_mesh


def simplify_mesh(mesh_path, max_faces=np.inf):
    mesh = trimesh.load(mesh_path, force='mesh', file_type='obj', resolver=None)

    obj_mesh = o3d.geometry.TriangleMesh()
    obj_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    obj_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    obj_mesh.remove_unreferenced_vertices()
    obj_mesh.compute_vertex_normals()

    if np.asarray(obj_mesh.triangles).shape[0] <= max_faces:
        obj_mesh.remove_unreferenced_vertices()
        obj_mesh.remove_degenerate_triangles()
        obj_mesh.remove_duplicated_triangles()
        obj_mesh.remove_duplicated_vertices()

        return obj_mesh

    else:
        voxel_size = max(obj_mesh.get_max_bound() - obj_mesh.get_min_bound()) / 64
        mesh_smp = obj_mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Quadric)

        mesh_smp.remove_unreferenced_vertices()
        mesh_smp.remove_degenerate_triangles()
        mesh_smp.remove_duplicated_triangles()
        mesh_smp.remove_duplicated_vertices()

        return mesh_smp


def preprocess_shapenet(shapenet_core_path, synset_id_core, shapenet_out_path, device):
    all_obj_list = os.listdir(os.path.join(shapenet_core_path, synset_id_core))

    for obj_id in all_obj_list:
        obj_path = os.path.join(shapenet_core_path, synset_id_core, obj_id, 'models', 'model_normalized.obj')
        if not os.path.exists(obj_path):
            continue

        out_path = os.path.join(shapenet_out_path, synset_id_core, obj_id, 'models')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_path_obj = os.path.join(out_path, 'model_normalized.obj')

        if os.path.exists(out_path_obj):
            continue

        try:
            mesh_o3d = simplify_mesh(obj_path)
        except:
            continue

        verts = torch.tensor(np.asarray(mesh_o3d.vertices)).float().to(device)
        faces = torch.tensor(np.asarray(mesh_o3d.triangles)).to(device)
        mesh_normalized, _ = normalize_mesh(verts, faces, device)

        if not torch.isfinite(mesh_normalized.verts_packed()).all():
            print("Meshes contain nan or inf.")
        else:
            IO().save_mesh(data=mesh_normalized, path=out_path_obj, include_textures=False)

    return


def main(args):
    # Setup
    if args.device == '':
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
    else:
        device = 'cuda:' + str(torch.cuda.current_device())

    shapenet_ScanNet_classes = ['trash bin', 'table', 'printer', 'basket', 'bookshelf', 'flowerpot',
                                'laptop', 'sofa', 'chair', 'file cabinet', 'display', 'bag', 'bathtub',
                                'cabinet', 'clock', 'bed', 'bench', 'stove', 'lamp', 'faucet', 'bowl',
                                'keyboard', 'piano', 'microwaves', 'dishwasher', 'washer', 'guitar', 'pillow',
                                'motorbike']

    shapenet_path = os.path.join(parent, 'data', 'ShapeNet')
    shapenet_core_path = os.path.join(shapenet_path, 'ShapeNetCore.v2')
    shapenet_out_path = os.path.join(shapenet_path, 'ShapeNet_preprocessed')

    if not os.path.exists(shapenet_out_path):
        os.makedirs(shapenet_out_path)

    for cls_name in shapenet_ScanNet_classes:
        print('Current Class = ' + str(cls_name))
        try:
            synset_id_core = shapenet_category_dict[cls_name]
        except:
            assert False

        preprocess_shapenet(shapenet_core_path, synset_id_core, shapenet_out_path, device)


if __name__ == "__main__":
    main(parser.parse_args())
