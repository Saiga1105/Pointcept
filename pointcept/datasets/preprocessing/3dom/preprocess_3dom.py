
import argparse
import os
import torch
from torch.serialization import save
import laspy
import numpy as np

from geomapi.utils import geometryutils as gmu
import open3d as o3d

def handle_process(file_name, output_folder):
    
    coords = []
    colors = []
    scene_id = os.path.basename(file_name)

    name, ext = os.path.splitext(scene_id)
    
    if ext not in  [".las", ".laz"]:
        return

    # Read LAS/LAZ
    # populate dict
    las = laspy.read(file_name)
    print(list(las.point_format.dimension_names))

    pcd = gmu.las_to_pcd(las)
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    
    coords = np.stack([las.x, las.y, las.z], axis=1)
    colors = np.stack([las.red, las.green, las.blue], axis=1) # betons
    # colors = np.zeros((len(las.points), 3)).astype(np.uint8) + 255
    # colors = np.stack([las.verticality * 255, las.omnivariance * 255, las.normal_change_rate * 255], axis=1).astype(np.uint8)
    normals = np.asarray(pcd.normals)    

    # save_dict = dict(coord=coords, color=colors, normal=normals, scene_id=scene_id, semantic_gt=las.classes.astype(int), instance_gt=las.objects.astype(int)) # beton
    save_dict = dict(coord=coords, color=colors, normal=normals, scene_id=scene_id, semantic_gt=las.labels.astype(int)) # gabry

    print(save_dict)

    torch.save(save_dict, os.path.join(output_folder, f"{name}.pth"))
    # asd = torch.load("/home/rha/roberto/projects/Pointcept/data/scannet/val/scene0063_00.pth")
    # print(asd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the 3DOM dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where processed files will be saved",
    )
    config = parser.parse_args()

    os.makedirs(config.output_root, exist_ok=True)

    for file_name in os.listdir(config.dataset_root):

        handle_process(os.path.join(config.dataset_root, file_name), config.output_root)