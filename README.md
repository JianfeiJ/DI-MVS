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
- Download pre-processed datasets for training: [dtu_training](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view), [Depth_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip).
  ```
  - dtu_training/
  - Cameras
  - Depths
  - Depths_raw
  - Rectified
  ```
### Tanks & Temples

Download [Tanks & Temples test dataset](https://drive.google.com/file/d/1gAfmeoGNEFl9dL4QcAU4kF0BAyTd-r8Z/view).
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
Download pretrained model on [DTU and BlendedMVS](https://drive.google.com/drive/folders/1BWPfXx4aPEjt6SsvvtZGNTriMTMHbHDp?usp=sharing), and remove them to the checkppints folder.
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
### Evaluation on Tanks & Temples
 ```
sh test_tnt.sh
 ```
### Result on DTU
|    Methods  |  Acc. (mm)     | Comp. (mm) | Overall (mm)   | Time (s)   |
|    :----:   |    :----:   |    :----:   |    :----:    |   :----:    |
| DI-MVS-lite     | 0.305      |0.305| 0.305   |  0.10  |
| DI-MVS      | 0.312      |0.278| 0.295|0.16|
### Result on [Tanks & Temples benchmark](https://www.tanksandtemples.org/leaderboard/AdvancedF/?table_0-sort=-my_mean).
|Intermediate|Advanced |
|    :----:   |    :----:   |
| 62.94      |40.92|

## Traning

 ```
sh train.sh
 ```

## Acknowledgements
This repository is partly based on [MVSNet](https://github.com/YoYo000/MVSNet), [Effi-MVS](https://github.com/bdwsq1996/Effi-MVS), [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet), [IterMVS](https://github.com/FangjinhuaWang/IterMVS).
Thanks for their excellent works!
