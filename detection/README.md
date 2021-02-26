# Detect Large TEM Images

## Usage

### Installation

#### Conda 

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov3-tf2-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov3-tf2-gpu
```

#### Nvidia Driver (For GPU)

```bash
# Ubuntu 18.04
sudo apt-add-repository -r ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-430
# Windows/Other
https://www.nvidia.com/Download/index.aspx
```

#### Download Weights

Version 1:

[Google Drive](https://drive.google.com/file/d/1AEGGwetIoUULm0kM1ZXKo6Zj_klLLMZt/view?usp=sharing)

md5: 49ff4e8e4d1235eee0408d431456ae6e

This version is trained on 80% data and selected on min validation loss. 

Version 2:

[Google Drive](https://drive.google.com/file/d/1eTTH9aRt8IJdYWjXzq_fbw6kqE2VvGMe/view?usp=sharing)

md5: 0c5972e0fdd6475f9671b669e733c952

This version is trained on 100% data and end on the 25 epochs. 


Please uzip and put the weights (three files) into `yolov3_tf2/`

### Files

| Name          | Description                     |
| ------------- | ------------------------------- |
| conda-cpu.yml | Conda environment file for cpu. |
| conda-gpu.yml | Conda environment file for gpu. |
| detect.py     | Detect image or images.         |
| output/       | Store detect results.           |
| samples/      | Store image samples.            |
| yolov3_tf2/   | Files for YoloV3 model.         |

### Detection

#### Important parameters

##### cut_size

The size of the sliding windows performed on the large image in pixels. Please set this value sush that each cut contains 4-6 particles in average. Empirically, this is the most important variable in detection. 

##### stride

The stride of the sliding window. For intuitive consideration, this value is stride/cut_size. 1/2 or 1/3 is recommended, smaller ones will cost more computation time with the same resutls. 

##### margin

This is the margin of image that will not be detected (this value is margin / cut_size). The default 1/16 is good enough for general cases. 

##### output_type

There are 5 types of output offered in the tool. 

- boxes: Output a txt file with basic bounding boxes in each line (x_min y_min x_max y_max)
- center: Output a txt file with center coordinates of boxes in each line (x_center y_center)
- center_size: Output a txt file with center coordinates of boxes and size in each line: (x_center y_center width*height)
- json: Output a json file that is compatible with labeling software "colabler"
- benchmark: Output a txt file for mAP calcuation ('particle' confidence x_min y_min x_max y_max)

##### image_directory & image_path

If the parameter "--image_path" is assigned to an image path, the program will only detect this single image. Otherwise, the program will detect all images (of image_type) in "--image_directory" .

##### image_type

The type of image files in "--image_directory". For example, "png", "tif", "jpeg", etc. 

##### Other parameters

Other parameters can be set to default.

#### Example 

```bash
python detect_img.py --cut_size 100 --image_type tif --image_directory samples/ --output_type boxes
```

## Command Line Args Reference

```bash
detect.py:
  --batch_size: batch size to detect
    (default: '12')
    (an integer)
  --classes: path to classes file
    (default: './yolov3_tf2/particle.names')
  --cut_size: resize images to
    (default: '100')
    (an integer)
  --dpi: dpi of output image
    (default: '300')
    (an integer)
  --image_directory: path to the directory of image files
    (default: './samples/')
  --image_path: path to image file
    (default: '')
  --image_type: file type of images
    (default: 'png')
  --margin: the margin of image that will not be detected (this value is margin / cut_size)
    (default: '0.0625')
    (a number)
  --num_classes: number of classes in the model
    (default: '1')
    (an integer)
  --output: output directory
    (default: './output/')
  --[no]output_image: whether output a result image marked with blue boxes
    (default: 'true')
  --output_type: <boxes|center|center_size|json|benchmark>: 
    boxes: bounding boxes: (x_min y_min x_max y_max),
    center: center coordinates: (x_center y_center), 
    center_size: center coordinates and size: (x_center y_center width*height), 
    json: .json file for future labeling in "colabler", 
    benchmark: for mAP calcuation: ('particle' confidence x_min y_min x_max y_max)
    (default: 'boxes')
  --size: resize images to
    (default: '416')
    (an integer)
  --stride: the stride of the sliding widow (this value is stride / cut_size)
    (default: '0.5')
    (a number)
  --weights: path to weights file
    (default: './yolov3_tf2/yolov3_model.tf')
```


## References

- https://github.com/zzh8829/yolov3-tf2
    - yolov3 implementation in tf2
