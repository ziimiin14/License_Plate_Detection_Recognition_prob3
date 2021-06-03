# License_Plate_Detection_Recognition_Pytorch
This repo is refers to the problem 3 in the technical assessment
1) [YOLOv5](https://github.com/ultralytics/yolov5) is being used for license plate detection. 
2) [LPRNet](https://arxiv.org/abs/1806.10447) is being used for license plate recognition (detect character on license plate).
3) [Spatial Transformer Layer](https://arxiv.org/abs/1506.02025) is embeded in this work to allow a better characteristics for recognition.

Tasks Completed:
1) [Training data] (https://www.kaggle.com/andrewmvd/car-plate-detection) for license plate detection.
2) [YOLOv5 training guide] (https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) for license plate detection.
3) YOLOv5s network which is a shrink network from YOLOv5 is being used for the purpose of lower inference time.
4) Combined YOLOv5 network together with LPRNet and STL to detect and recognize license plate.


Results

<img src="test/SGcar_1.jpg"  width="450" style="float: left;"> <img src="test_result/detected_SGcar_1.jpg"  width="450" style="float: left;">


## LPRNet Performance 
LPRNet coding is heavily followed by [sirius-ai](https://github.com/sirius-ai/LPRNet_Pytorch)'s repo. One exception is that the spatial transformer layer is inserted to increase the accuracy reported on CCPD database as below: 

|   | Base(45k) | DB | FN | Rotate | Tilt | Weather | Challenge |
|  :------:     | :---------: | :---------: |:---------: |:---------: |:---------: |:---------: |:---------: |
|   accuracy %      | 99.1     |  96.3 | 97.3 | 95.1 | 96.4 | 97.1 | 83.2 |

## Training on MTCNN
* Download the [CCPD](https://github.com/detectRecog/CCPD) data and put it into 'ccpd' folder
* run 'MTCNN/data_set/preprocess.py' to split training data and validation data and put in "ccpd_train" and "ccpd_val" folders respectively.
* run 'MTCNN/data_preprocessing/gen_Pnet_train_data.py', 'MTCNN/data_preprocessing/gen_Onet_train_data.py','MTCNN/data_preprocessing/assemble_Pnet_imglist.py', 'MTCNN/data_preprocessing/assemble_Onet_imglist.py' for training data preparation.
* run 'MTCNN/train/Train_Pnet.py' and 'MTCNN/train/Train_Onet.py

## Training on LPRNet
* run 'LPRNet/data/preprocess.py' to prepare the dataset
* run 'LPRNet/LPRNet_Train.py' for training 

## Test
* run 'MTCNN/MTCNN.py' for license plate detection
* run 'LPRNet/LPRNet_Test.py' for license plate recognition
* run 'main.py' for both

## Reference
* [MTCNN](https://arxiv.org/abs/1604.02878v1)
* [LPRNet](https://arxiv.org/abs/1806.10447)
* [Spatial Transformer Layer](https://arxiv.org/abs/1506.02025)
* [LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)

**Please give me a star if it is helpful for your research**
