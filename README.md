# RIFE Recursive Image Interpolation

This repository implements image frame interpolation using the RIFE (Recursive Image Flow Estimation) model. It provides a solution for video frame interpolation with an optional recursive strategy to improve the interpolation quality. The model utilizes a trained RIFE model (RIFEv4-25) to generate intermediate frames between a pair of input images.

## Features

* Interpolate between frames in a video or image sequence.
* Supports UHD (4K) image interpolation.
* Option to use recursive inference for improved quality.
* Scalable options for different output resolutions.
* Supports multiple image formats (`.png`, `.tif`, `.jpg`, `.jpeg`, `.bmp`, `.gif`).

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Recursive Inference](#recursive-inference)
6. [Input and Output](#input-and-output)
7. [Model Files](#model-files)
8. [License](#license)

---

## Overview

The RIFE Recursive Image Interpolation model generates intermediate frames by interpolating between consecutive frames in a sequence. This is useful for applications such as:

* Slow-motion video creation.
* Frame interpolation for higher frame rates.
* Animation or visual effects work.

The model has been enhanced by incorporating recursive strategies to improve the quality of interpolated frames, ensuring smoother transitions and reducing artifacts.

---

## Requirements

To run the project, you will need the following dependencies:

* Python 3.6+
* PyTorch (preferably CUDA-enabled for GPU support)
* OpenCV
* NumPy
* tqdm (for progress bars)

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

---

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/mehdinejjar86/recursive_rife.git
cd recursive_rife
```

Ensure you have the required trained RIFE model files, which are necessary for inference. These models should be placed in the `rifev4_25` directory (or the directory you specify in the arguments). You can download the required model (`flownet.pkl`) from the original RIFE repository.

Here is the structure of your repository once everything is set up:

```
recursive_rife/
├── ckpt/
│   └── rifev4_25/
│       └── flownet.pkl
│       └── (other_model_files.py)
├── inference.py
├── (other project files)
└── README.md
```

* Download the `flownet.pkl` file from the [original RIFE repository](https://github.com/hzwer/arXiv2020-RIFE). This file is necessary for the frame interpolation process.
* Place `flownet.pkl` inside the `recursive_rife/ckpt/rifev4_25/` directory.

---

## Usage

### Command-Line Arguments

To run the script, use the following command:

```bash
python interpolate.py --model <model_directory> --img <input_images_directory> --output <output_directory> [--UHD] [--recursive] [--scale <scale>] [--multi <multi>]
```

### Arguments:

* `--model`: Directory containing the trained model files (default is `rifev4_25`).
* `--UHD`: Enable UHD (4K) support, reducing scale to 0.5 automatically.
* `--scale`: Adjust the scaling factor for interpolation (options: 0.25, 0.5, 1.0, 2.0, 4.0).
* `--multi`: Controls the number of interpolated frames between two images (default is `2`).
* `--img`: Input directory containing the image sequence or frames to interpolate.
* `--output`: Output directory where the interpolated frames will be saved.
* `--recursive`: Use recursive inference for improved interpolation quality.

### Example Command:

```bash
python interpolate.py --model ckpt/rifev4_25 --img ./input_frames --output ./output_frames --recursive --scale 1.0 --multi 2
```

This will interpolate frames from the images in `./input_frames` directory, saving the interpolated frames in `./output_frames`.

---

## Recursive Inference

The script provides an option to use recursive inference, which can produce smoother and more accurate interpolations by applying the interpolation model iteratively. This is controlled by the `--recursive` flag. When enabled, the model recursively refines the interpolated results, resulting in better quality especially in high-resolution video or images.

### How It Works:

* **Standard Inference**: The model generates intermediate frames between two consecutive images in a sequence.
* **Recursive Inference**: The model refines these intermediate frames by applying the interpolation process recursively. This enhances the smoothness and quality of transitions.

---

## Input and Output

### Input:

The input sequence should consist of images with even-numbered filenames (e.g., `0.png`, `2.png`, `4.png`, etc.). The script assumes these images are provided in sequential order, with each image representing a key frame in the video or animation sequence.

### Output:

The output will be a sequence of interpolated frames, with the intermediate frames inserted between the even-numbered frames. For example, if the input sequence is:

* `0.png`, `2.png`, `4.png`, ...

The output sequence will include:

* `0.png`, `1.png` (interpolated), `2.png`, `3.png` (interpolated), `4.png`, ...

The interpolated frames will be saved in the specified output directory, following the naming convention of the original frames.

### Example:

**Input**:

```
0.png 2.png 4.png 6.png
```

**Output**:

```
0.png 1.png 2.png 3.png 4.png 5.png 6.png
```

The output will have interpolated frames (`1.png`, `3.png`, `5.png`) placed between the existing frames (`0.png`, `2.png`, `4.png`, `6.png`).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
