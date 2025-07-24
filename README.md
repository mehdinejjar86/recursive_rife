# Practical-RIFE-Interpolation

This repository provides a practical and efficient solution for video frame interpolation utilizing the RIFE (Real-time Intermediate Flow Estimation) model. It enables users to generate recursivly smooth, high-quality intermediate frames between existing video frames, effectively increasing video frame rates, creating slow-motion effects, or enhancing the temporal consistency of generated video sequences.

## 🚀 Getting Started

Follow these steps to set up and run the RIFE interpolation script on your local machine.

### Prerequisites

Ensure you have the following installed:

- Python (3.8 - 3.11 recommended)
- PyTorch + Also Support Metal Performance Shaders

### Installation

Clone the repository to your local machine:

```bash
git clone git@github.com:mehdinejjar86/recursive_rife.git
cd recursive_rife
pip install -r requirements.txt
```

### Download Trained Model

You will need a pre-trained RIFE model checkpoint.

1.  **Download:** Download the `model.pkl` file from the links provided below in the "Supported Models" section.
2.  **Placement:** Place the downloaded `model.pkl` (only) into this subdirectory.
    - Example structure:
      ```
      recursive_rife/
      ├── ckpt/
      │   └── rifev4_25/
      │       └── flownet.pkl
      │       └── (any_other_model_files.py)
      ├── inference.py
      ├── (other project files)
      └── README.md
      ```

## 🛠 Usage

The `inference_img.py` script is the main entry point for performing interpolation. It processes a sequence of images from an input directory, interpolating between each consecutive pair, and saving the results to an output directory.

### Command-line Arguments

- `--model`: **Required.** Path to the directory containing the trained RIFE model files (e.g., `ckpt/rifev4_25`).
- `--img`: **Required.** Path to the input directory containing your source images. Images should be named numerically and sequentially (e.g., `0000000.png`, `0000001.png`, `0000002.png`).
- `--output`: **Optional.** Path to the directory where interpolated frames will be saved. If not specified, defaults to `vid_out/`.
- `--multi`: **Crucial Parameter.** An integer defining the **total number of frames to output for each original input frame pair**, _including_ the two original frames themselves.
  - `--multi 3`: Inserts `2` frame between each original pair.
  - `--multi 5`: Inserts `4` frames between each original pair.
  - `--multi N`: Inserts `N-1` frames between each original pair.
- `--UHD`: **Flag.** Use this flag to enable support for 4K images. When set, `--scale` automatically defaults to `0.5` unless explicitly overridden.
- `--scale`: **Optional.** A float specifying the internal scaling factor for model processing. Must be one of `0.25`, `0.5`, `1.0`, `2.0`, or `4.0`. (Default: `1.0`). Use `0.5` or `0.25` for very high-resolution inputs to manage memory.
- `--recursive`: **Flag.** Use this flag to enable the experimental recursive interpolation mode. See "Interpolation Modes" below for detailed explanation. (Default: `False`).

### Example Run

To interpolate `15` frames between each pair of images in `./data/input_frames/` using the `rifev4_25` model, saving to `./data/output_frames/`:

```bash
python inference_img.py --model ckpt/rifev4_25 --multi 16 --img ./data/input_frames --output ./data/output_frames
```

---

## Interpolation Modes

This project offers two distinct interpolation modes: the standard direct approach and an advanced recursive method. It's important to understand their differences to choose the most suitable one for your needs and model version.

### 1\. Standard Interpolation (Recommended)

- **How to use:** Do **NOT** include the `--recursive` flag in your command. This is the default behavior.
- **Underlying Function:** `make_inference`
- **Mechanism:** This mode directly utilizes the `timestep` parameter of the RIFE model's `inference` function. For a given `--multi N`, it calculates `N-2` intermediate frames by evenly distributing the `timestep` values across the `0` to `1` interval (e.g., `1/(N-1)`, `2/(N-1)`, ..., `(N-2)/(N-1)`).
- **When to use:** This is the **recommended mode** for most modern RIFE models (especially versions 3.9 and higher, like `rifev4_25`). These models are typically trained to perform robustly across a continuous range of `timestep` values, providing the most flexible and consistent results for any `N` specified by `--multi`.

### 2\. Recursive Interpolation (Advanced / Experimental)

- **How to use:** Include the `--recursive` flag in your command (e.g., `python inference_img.py --recursive ...`).
- **Underlying Function:** `make_inference_recursive` (for `model.version >= 3.9`)
- **Mechanism:** This mode implements a more complex, multi-stage iterative refinement strategy. Instead of a single direct inference per intermediate frame, it performs several passes, incrementally refining flow and mask predictions for a sequence of frames. It's designed to build up the interpolated sequence through a series of specialized inferences.
- **When to use:** This mode might be considered for:
  - **Exploring alternative strategies:** If you are conducting research or experiments into different methods of flow-based interpolation refinement.
  - **Specific model behaviors:** In rare cases, some RIFE versions or custom-trained models might exhibit unique characteristics where iterative refinement yields marginally better results for specific content, although this is less common with general-purpose modern RIFE models.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
