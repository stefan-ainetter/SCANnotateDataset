import argparse
import copy
import os
import sys

parser = argparse.ArgumentParser(description="Code for visualization of scannotate annotations for ScanNet")
parser.add_argument("--device", type=str, default="", help="device")

os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().device

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)

import pickle
import torch
import numpy as np
import open3d as o3d
from load_ScanNet_data import load_axis_alignment_mat
from misc.utils import transform_ScanNet_to_py3D, alignPclMesh
from misc.utils import COLOR_DETECTRON2
from misc.utils_CAD_retrieval import load_textured_cad_model_prepro, drawOpen3dCylLines
from ScanNetAnnotation import *


def load_scene_mesh(meta_file_path, scene_path):
    mesh_o3d = o3d.io.read_triangle_mesh(scene_path)

    # Transfer points to py3d coord system
    T_mat = transform_ScanNet_to_py3D()

    align_mat_Scannet = load_axis_alignment_mat(meta_file_path=meta_file_path)
    align_mat_Scannet = np.reshape(np.asarray(align_mat_Scannet), (4, 4))

    mesh_o3d = alignPclMesh(mesh_o3d, axis_align_matrix=align_mat_Scannet, T=T_mat)

    return mesh_o3d


def visualize_annotations(scene_obj, shapenet_preprocessed_path, device):
    obj_mesh_all = None
    lineSets_all = None
    for box_item in scene_obj.obj_annotation_list:

        model_path = os.path.join(shapenet_preprocessed_path, box_item.catid_cad, box_item.id_cad, 'models',
                                  'model_normalized.obj')
        cad_transform_base = box_item.transform3d.to(device)

        cad_model_o3d = load_textured_cad_model_prepro(model_path, cad_transform_base, box_item.category_label, device)

        if obj_mesh_all is None:
            obj_mesh_all = cad_model_o3d
        else:
            obj_mesh_all += cad_model_o3d

        x_shift = .5
        y_shift = .5
        z_shift = .5

        box = np.array([[x_shift, y_shift, z_shift], [-x_shift, y_shift, z_shift], [-x_shift, -y_shift, z_shift],
                        [x_shift, -y_shift, z_shift], [x_shift, y_shift, -z_shift], [-x_shift, y_shift, -z_shift],
                        [-x_shift, -y_shift, -z_shift], [x_shift, -y_shift, -z_shift]])

        box_tensor = torch.Tensor(box).to(device)

        box_transformed = cad_transform_base.transform_points(box_tensor)
        box_transformed = box_transformed.cpu().detach().numpy()
        if box_item.is_in_scan2cad:
            line_color = [0, 0, 1]
        else:
            line_color = [1, 0, 0]

        lineSets = drawOpen3dCylLines([box_transformed], line_color)

        if lineSets_all is None:
            lineSets_all = lineSets
        else:
            lineSets_all += lineSets

    return obj_mesh_all, lineSets_all


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

    scannet_path = os.path.join(current_dir, 'data', 'ScanNet')
    annotations_path = os.path.join(scannet_path, 'annotations')
    scans_path = os.path.join(scannet_path, 'scans')
    shapenet_path = os.path.join(current_dir, 'data', 'ShapeNet', 'ShapeNet_preprocessed')

    scene_list = os.listdir(annotations_path)

    for scene_name in scene_list:

        meta_file_path = os.path.join(scans_path, scene_name,
                                      scene_name + '.txt')

        scene_path = os.path.join(scans_path, scene_name,
                                  scene_name + '_vh_clean_2.ply')

        annotation_file = os.path.join(annotations_path, scene_name, scene_name + '.pkl')

        if not os.path.exists(meta_file_path) or not os.path.exists(scene_path) or not os.path.exists(annotation_file):
            print('Data not available for scene ' + str(scene_name))
            continue

        scene_mesh = load_scene_mesh(meta_file_path, scene_path)

        pkl_file = open(annotation_file, 'rb')
        scene_obj = pickle.load(pkl_file)

        obj_mesh_all, lineSets_all = visualize_annotations(scene_obj, shapenet_path, device)

        final_inst_seg_3d = scene_obj.inst_seg_3d
        points_mesh = copy.deepcopy(np.asarray(scene_mesh.vertices))
        pcl_color_tmp = np.zeros((points_mesh.shape[0], 3))
        for color_id in np.unique(final_inst_seg_3d):
            if color_id < 1.:
                continue
            pcl_color_tmp[final_inst_seg_3d == int(color_id)] = COLOR_DETECTRON2[int(color_id), :]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_mesh)
        pcd.colors = o3d.utility.Vector3dVector(pcl_color_tmp)

        indices = np.where(final_inst_seg_3d > 0)[0].tolist()
        if not indices:
            continue

        mesh_o3d = copy.deepcopy(scene_mesh)
        mesh_o3d.remove_vertices_by_index(indices)
        bg_and_anno = obj_mesh_all + mesh_o3d + lineSets_all

        # Visualize
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Open3D', width=640 * 2, height=480 * 2, left=100, top=200,
                          visible=True)
        vis.get_render_option().mesh_show_back_face = False

        sceneMeshVert = np.array(scene_mesh.vertices)
        trans = np.zeros((3,)) * 1.2
        trans[0] = np.max(sceneMeshVert[:, 0]) - np.min(sceneMeshVert[:, 0])
        vis.add_geometry(bg_and_anno)

        vis.add_geometry(pcd.translate(trans * 1.2))
        scene_and_boxes = scene_mesh + lineSets_all
        vis.add_geometry(scene_and_boxes.translate(trans * 1.2 * 2))
        vis.run()


if __name__ == "__main__":
    main(parser.parse_args())
