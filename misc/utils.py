import numpy as np
import open3d as o3d

COLOR_DETECTRON2 = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.000, 0.667, 0.000,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.000, 0.000, 1.000,
        0.000, 1.000, 0.000,
        0.749, 0.749, 0.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        0.200, 0.733, 1.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        # 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        # 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        # 0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        # 0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        # 0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        # 1.000, 1.000, 1.000
    ]).astype(np.float32).reshape(-1, 3)

SEMANTIC_IDXS = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
     30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])

SEMANTIC_NAMES = np.array(['unannotated', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                           'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser',
                           'pillow',
                           'mirror', 'floor_mat', 'clothes', 'ceiling', 'book', 'refridgerator', 'television', 'paper',
                           'towel',
                           'shower curtain', 'box', 'whiteboard', 'person', 'night stand',
                           'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop'])

SEMANTIC_IDX2NAME = {0: 'unannotated', 1: 'wall', 2: 'floor', 3: 'cabinet', 4: 'bed', 5: 'chair', 6: 'sofa',
                     7: 'table', 8: 'door', 9: 'window', 10: 'bookshelf', 11: 'picture',
                     12: 'counter', 13: 'blinds', 14: 'desk', 15: 'shelves', 16: 'curtain', 17: 'dresser', 18: 'pillow',
                     19: 'mirror', 20: 'floor_mat', 21: 'clothes', 22: 'ceiling', 23: 'book', 24: 'refridgerator',
                     25: 'television', 26: 'paper', 27: 'towel', 28: 'shower curtain', 29: 'box', 30: 'whiteboard',
                     31: 'person', 32: 'night stand', 33: 'toilet', 34: 'sink', 35: 'lamp', 36: 'bathtub', 37: 'bag',
                     38: 'otherstructure', 39: 'otherfurniture', 40: 'otherprop'}

shapenet_category_dict = {'airplane': '02691156', 'trash bin': '02747177', 'bag': '02773838', 'basket': '02801938',
                          'bathtub': '02808440', 'bed': '02818832', 'bench': '02828884', 'birdhouse': '02843684',
                          'bookshelf': '02871439', 'bottle': '02876657', 'bowl': '02880940', 'bus': '02924116',
                          'cabinet': '02933112', 'camera': '02942699', 'can': '02946921', 'cap': '02954340',
                          'car': '02958343', 'cellphone': '02992529', 'chair': '03001627', 'clock': '03046257',
                          'keyboard': '03085013', 'dishwasher': '03207941', 'display': '03211117',
                          'earphone': '03261776', 'faucet': '03325088', 'file cabinet': '03337140',
                          'guitar': '03467517',
                          'helmet': '03513137', 'jar': '03593526', 'knife': '03624134', 'lamp': '03636649',
                          'laptop': '03642806', 'loudspeaker': '03691459', 'mailbox': '03710193',
                          'microphone': '03759954', 'microwaves': '03761084', 'motorbike': '03790512',
                          'mug': '03797390', 'piano': '03928116', 'pillow': '03938244', 'pistol': '03948459',
                          'flowerpot': '03991062', 'printer': '04004475', 'remote': '04074963', 'rifle': '04090263',
                          'rocket': '04099429', 'skateboard': '04225987', 'sofa': '04256520', 'stove': '04330267',
                          'table': '04379243', 'telephone': '04401088', 'tower': '04460130', 'train': '04468005',
                          'watercraft': '04530566', 'washer': '04554684', 'desk': '03179701', 'dresser': '03237340',
                          'bed cabinet': '20000008'}


def Ry(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def Rx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def Rz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def alignPy3D_format(pclMesh, T_mat):
    verts = np.array(pclMesh.verts_list()[0])
    newVerts = np.ones((verts.shape[0], 4))
    newVerts[:, :3] = verts
    newVerts = newVerts.dot(T_mat.T)
    pclMesh.vertices = o3d.utility.Vector3dVector(newVerts[:, :3])
    return pclMesh


def transform_ScanNet_to_py3D():
    rot_tmp1 = Rx(np.deg2rad(-90))
    rot_tmp2 = Ry(np.deg2rad(-90))
    rot3 = np.asarray(np.dot(rot_tmp2, rot_tmp1))
    T = np.eye(4)
    T[:3, :3] = rot3
    return T


def transform_ARKIT_to_py3D():
    rot_tmp1 = Rx(np.deg2rad(-90))
    rot_tmp2 = Ry(np.deg2rad(-180))
    rot3 = np.asarray(np.dot(rot_tmp2, rot_tmp1))
    T = np.eye(4)
    T[:3, :3] = rot3
    return T


def alignPclMesh(pclMesh, axis_align_matrix=np.eye(4), T=np.eye(4)):
    if isinstance(pclMesh, o3d.geometry.TriangleMesh):
        verts = np.array(pclMesh.vertices)
        newVerts = np.ones((verts.shape[0], 4))
        newVerts[:, :3] = verts
        newVerts = newVerts.dot(axis_align_matrix.T)
        newVerts = newVerts.dot(T.T)
        pclMesh.vertices = o3d.utility.Vector3dVector(newVerts[:, :3])

    elif isinstance(pclMesh, o3d.geometry.PointCloud):
        points = np.array(pclMesh.points)
        newPoints = np.ones((points.shape[0], 4))
        newPoints[:, :3] = points
        newPoints = newPoints.dot(axis_align_matrix.T)
        newPoints = newPoints.dot(T.T)
        pclMesh.points = o3d.utility.Vector3dVector(newPoints[:, :3])

    elif isinstance(pclMesh, np.ndarray):
        points = pclMesh
        newPoints = np.ones((points.shape[0], 4))
        newPoints[:, :3] = points
        newPoints = newPoints.dot(axis_align_matrix.T)
        newPoints = newPoints.dot(T.T)
        return newPoints

    else:
        raise NotImplementedError

    return pclMesh
