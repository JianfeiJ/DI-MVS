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
  - dtu/
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

Download [Tanks & Temples test dataset](https://drive.google.com/file/d/1gAfmeoGNEFl9dL4QcAU4kF0BAyTd-r8Z/view)
 ```
- tanksandtemples_1/
  - advanced
    - ...
    - Temple
      - cams
      - images
      - pair.txt
      - Temple.log
  - intermediate
    - ...
    - Train
      - cams
      - cams_train
      - images
      - pair.txt
      - Train.log
 ```

## Reproducing Results
Download pretrained model on [DTU and BlendedMVS](https://drive.google.com/drive/folders/1BWPfXx4aPEjt6SsvvtZGNTriMTMHbHDp?usp=sharing), and remove them to checkppints folder.
### Evaluation on DTU
 ```
sh test_dtu.sh
 ```
For quantitative evaluation on DTU dataset, download [SampleSet](http://roboimagedata.compute.dtu.dk/?page_id=36) and [Points](http://roboimagedata.compute.dtu.dk/?page_id=36). Unzip them and place Points folder in SampleSet/MVS Data/.
```
- SampleSet
  - MVS Data
    - ObsMask
    - Points
```
### Tanks & Temples
 ```
sh test_tnt.sh
 ```
### Result on DTU
|    Methods  |  Acc. (mm)     | Comp. (mm) | Overall (mm)   |
|    :----:   |    :----:   |    :----:   |    :----:    |
| PatchmatchNet (1600×1200)      | 0.427      |0.277| 0.352   |
| Ours (1152×864)      | 0.442      |0.262| 0.352   |
| Ours (1600×1152)      | 0.427      |0.256| 0.342   |
### Result on Tanks & Temples
|    Training Dataset  |Intermediate|Advanced |
|    :----:   |    :----:   |    :----:   |
| DTU      | 56.70|35.59|
| BlendedMVS      | 57.67      |35.89|
