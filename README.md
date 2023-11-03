# Semantic Mapping
(1) This work contains the repos of a semantic segmentation project from CSAILVision at https://github.com/CSAILVision/semantic-segmentation-pytorch
(2) Collecting Data from habitat simulation, using these data as dataset for semantic model, after training, reconstruct the semantic map by the collected data

## Preparation
(1) Set up pytorch, Cuda and Cudnn 
(2) Get the following repos from https://github.com/CSAILVision/semantic-segmentation-pytorch 
(3) gitclone git@github.com:CSAILVision/semantic-segmentation-pytorch.git
(4) overwrite the codes in semantic-segmentation-pytorch directory with the codes in the same name directory in THIS REPOS, some changes has been modified for single GPU training and calculating mean IoU

## Hardware Configuration
(1) CPU: Intel 8700
(2) GPU: Nvidia GeForce GTX 1060 6GB

## Environment Configuration
(1) Pytorch version: 1.13.0
(2) Cuda: 11.7
(3) Cudnn: 8.6.0
(4) python: 3.7 (for habitat simulation, you may use conda to create an environment)

## Collecting Data
(1) Suppose you have ALREADY install Habitat environment and ALREADY DOWNLOAD the relpica dataset, and all the file path are on set.
(2) Run data_generator.py to collect data
(3) Run load.py to collect the data for reconstruction
(4) Generating odgt file
(5) For tranning dataset, modify the file structure  so it should be like:
    -dataset/
      -annotations/
        -training/
          -validation/
      -images/
        -training/
        -validation/
(6) For the data collected by load.py, modify the file structure  so it should be like: 
    -first_floor/ 
      -annotations/
        -validation/
      -images/
        -validation
        
## Model Training
(1) Select one model architecture and download the pre-trained checkpoint
(2) Customize your own configuration file. Make sure the data path and all the parameters are set correctly.
(3) Run train.py to train two semantic segmentation models. One is trained on images collected from other scenes, the other one is trained on images collected from apartment_0
(4) The checkpoint share link: https://drive.google.com/drive/folders/1sIP8_SW8VgBwEH7qAzY8Vczi8KmDgAn0?usp=drive_link
## 3D Semantic Map
(1) To reconstruct the scene, execute the "3d_semantic_map.py" code. While the program is running, it will provide updates on the total number of pictures being processed, including the currently processed picture. Additionally, it will display the time taken for both global registration and ICP for each picture. 

(2) Users can customize the reconstruction process by providing arguments to the code. For instance, to reconstruct the first floor using the open3d ICP method, you can run the following command in the terminal:

(3) $ python 3d_semantic_map.py -f 1 -v open3d 
(This command allows you to specify the floor level ("-f 1") and the version of ICP to use ("-v my_icp") for the reconstruction process.)

## Results

