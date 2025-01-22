# License-Plate-Recognition

This repository is a reproduction of the paper [A Robust Real-Time Automatic License Plate Recognition Based on the YOLO Detector](https://ieeexplore.ieee.org/document/8489629). Since a lot of detail is missing in the paper, we could only guess about configuration settings of the models and image processing steps. We also had to make some changed to be able to run up-to-date CUDA. This repository automatically generates the training data for all five models and allows you to call each models or the entire model pipeline.

> This repository does not include the trainingdata since it is not publicly available. You can request the dataset from the original authors [here](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/license-agreement/).

Our model results are as follows:
<table>
  <thead>
    <tr>
      <th></th>
      <th>Recall / Accuracy (Our Results)</th>
      <th>Recall / Accuracy (Paper Results)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Vehicle Detection</td>
      <td>97.60%</td>
      <td>100%</td>
    </tr>
    <tr>
      <td>License Plate Detection</td>
      <td>94.88%</td>
      <td>98.33%</td>
    </tr>
    <tr>
      <td>Character Segmentation</td>
      <td>94.36%</td>
      <td>95.97%</td>
    </tr>
    <tr>
      <td>Character Recognition</td>
      <td>
        Digit: 83.87%<br>
        Letter: 88.08%<br>
        Avg: 85.68%
      </td>
      <td>
        Digit: 90.37%
      </td>
    </tr>
    <tr>
      <td>Entire Pipeline</td>
      <td>61.83%</td>
      <td>64.89%</td>
    </tr>
    <tr>
      <td>Entire Pipeline with Temporal Redundancy</td>
      <td>73.33%</td>
      <td>78.33%</td>
    </tr>
  </tbody>
</table>



## Setup
- We used WSL2 with Python 3.12.3
- Install darknet as described [here](#darknet-installation)
- install dependencies with `pip install -r requirements.txt`
- Configure `model_directory` and `ufpr_alpr_dirctory` in `settings.json` to point to where the models should be saved to, and where the UFPR-ALPR dataset can be found.
- Generate the datasets:
    - `python generate_vehicle_dataset.py`
    - `python generate_licenseplate_dataset.py`
    - `python generate_character_segmentation_dataset.py`
    - `python generate_character_recognition_digit_dataset.py`
    - `python generate_character_recognition_letter_dataset.py`
- Train each model with the command that is printed during the generate dataset step

## Usage
- Run `python inference_detector <model_name> <path_to_img>` to detect on a single model
- Run `python pipeline <path_to_image>` to detect the entire pipeline on a single image
- Run `python pipeline <path_to_folder_of_images>` to detect the entire pipeline on a series of images with temporal redundancy

## Darknet Installation
### Install cuda
- `wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin`
- `sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600`
- `wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.de`
- `sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb`
- `sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/`
- `sudo apt-get update`
- `sudo apt-get -y install cuda-toolkit-12-6`

### Install cudnn
- `wget https://developer.download.nvidia.com/compute/cudnn/9.5.1/local_installers/cudnn-local-repo-ubuntu2204-9.5.1_1.0-1_amd64.deb`
- `sudo dpkg -i cudnn-local-repo-ubuntu2204-9.5.1_1.0-1_amd64.deb`
- `sudo cp /var/cudnn-local-repo-ubuntu2204-9.5.1/cudnn-*-keyring.gpg /usr/share/keyrings/`
- `sudo apt-get update`
- `sudo apt-get -y install cudnn-cuda-12`

### Add cuda to PATH
- `export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}`


### Darknet
- `sudo apt-get install build-essential git libopencv-dev cmake`
- `mkdir ~/src`
- `cd ~/src`
- `git clone -b v2 https://github.com/hank-ai/darknet`
- `cd darknet`
- `mkdir build`
- `cd build`
- `cmake -DCMAKE_BUILD_TYPE=Release ..`
- `make -j4 package`
- `sudo dpkg -i darknet-2.1.2-Linux.deb`