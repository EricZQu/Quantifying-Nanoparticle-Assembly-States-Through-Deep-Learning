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





#### Example 

```bash
python train.py --classes ./data/particle.names --dataset ./data/particle_train.tfrecord --val_dataset ./data/particle_val.tfrecord --epochs 25 --learning_rate 1e-4 --num_classes 1 --transfer darknet --weights ./checkpoints/yolov3.tf --weights_num_classes 80
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
