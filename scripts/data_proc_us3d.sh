# UrbanScene3D, Residence, Sci-Art
python tools/copy_images.py --image_path data/urban_scene_3d/Residence/photos --dataset_path data/urban_scene_3d/residence-pixsfm
python tools/copy_images.py --image_path data/urban_scene_3d/Sci-Art/photos --dataset_path data/urban_scene_3d/sci-art-pixsfm

mv data/colmap_results/residence/train/sparse data/urban_scene_3d/residence-pixsfm/train
mv data/colmap_results/residence/val/sparse data/urban_scene_3d/residence-pixsfm/val

mv data/colmap_results/sciart/train/sparse data/urban_scene_3d/sci-art-pixsfm/train
mv data/colmap_results/sciart/val/sparse data/urban_scene_3d/sci-art-pixsfm/val
