# DI-MVS
## Installation
```
git clone https://github.com/JianfeiJ/DI-MVS.git
cd DI-MVS
```
## Enviorment
```
conda create -n dimvs python=3.8
conda activate dimvs
pip install -r requirements.txt
```
## Datasets
### DTU
- Download pre-processed datasets for test: [dtu](https://drive.google.com/file/d/1jN8yEQX0a-S22XwUjISM8xSJD39pFLL_/view).
  ```
  - dtu
  - scan1 (scene_name1)
  - scan2 (scene_name2)
    - images
      - 00000000.jpg
      - 00000001.jpg
      - ...
    - cams_1
      - 00000000_cam.txt
      - 00000001_cam.txt
      - ...
    - pair.txt
  ```
- Download pre-processed datasets for training: [dtu_training](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view), [Depth_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip)
  ```
  - dtu_training/
  - Cameras
  - Depths
  - Depths_raw
  - Rectified
  ```
### Tanks & Temples

Download [Tanks&Temples](https://drive.google.com/file/d/1gAfmeoGNEFl9dL4QcAU4kF0BAyTd-r8Z/view) for evaluation on 

## Test
