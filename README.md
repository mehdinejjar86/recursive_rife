# Practical-RIFE-Interpolation

<p align="center">
  <img src="image/grid.png" alt="Sample Interpolation Grid" width="600"/>
</p>

This repository provides a practical and efficient solution for video frame interpolation utilizing the RIFE (Real-time Intermediate Flow Estimation) model. It enables users to generate smooth, high-quality intermediate frames between existing video frames, effectively increasing video frame rates, creating slow-motion effects, or enhancing the temporal consistency of generated video sequences (e.g., from diffusion models).

## ✨ Key Features

- **RIFE Model Integration:** Built upon the high-performance RIFE model for state-of-the-art frame interpolation.
- **Flexible Frame Insertion:** Capable of generating an arbitrary number of intermediate frames between any two consecutive input frames.
- **Dynamic Resolution Handling:** Supports various input image resolutions, with optional downscaling (`--scale`) for optimizing performance on UHD (4K) content.
- **Multi-GPU Acceleration:** Leverages PyTorch for accelerated inference, enabling faster processing on compatible GPU hardware.
- **Efficient I/O Pipeline:** Implements a multi-threaded architecture with queues for asynchronous image reading and writing, mitigating I/O bottlenecks and reducing the risk of Out-of-Memory (OOM) errors during large-scale processing.
- **Intelligent Static Frame Skipping:** Automatically detects and skips interpolation for highly similar (static) consecutive frames based on SSIM, optimizing processing time.
- **Progress Tracking:** Integrates `tqdm` progress bars to provide real-time feedback on image reading, interpolation, and writing stages.
- **Dual Interpolation Modes:** Offers both a standard direct interpolation method and an advanced recursive approach for specific use cases (see "Interpolation Modes" below).

## 🚀 Getting Started

Follow these steps to set up and run the RIFE interpolation script on your local machine.

### Prerequisites

Ensure you have the following installed:

- Python (3.8 - 3.11 recommended)
- PyTorch + Also Support Metal Performance Shaders

### Installation

Clone the repository to your local machine:

```bash
git clone [https://github.com/mehdinejjar86/recursive_rife.git](https://github.com/mehdinejjar86/recursive_rife.git)
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
      │       └── model.pkl
      │       └── (any_other_model_files.py)
      ├── inference_img.py
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
  - `--multi 2`: Outputs only the original frames (no interpolation).
  - `--multi 3`: Inserts `1` frame between each original pair.
  - `--multi 5`: Inserts `3` frames between each original pair.
  - `--multi N`: Inserts `N-2` frames between each original pair.
- `--UHD`: **Flag.** Use this flag to enable support for 4K images. When set, `--scale` automatically defaults to `0.5` unless explicitly overridden.
- `--scale`: **Optional.** A float specifying the internal scaling factor for model processing. Must be one of `0.25`, `0.5`, `1.0`, `2.0`, or `4.0`. (Default: `1.0`). Use `0.5` or `0.25` for very high-resolution inputs to manage memory.
- `--recursive`: **Flag.** Use this flag to enable the experimental recursive interpolation mode. See "Interpolation Modes" below for detailed explanation. (Default: `False`).

### Example Run

To interpolate `15` frames between each pair of images in `./data/input_frames/` using the `rifev4_25` model, saving to `./data/output_frames/`:

```bash
python inference_img.py --model ckpt/rifev4_25 --multi 17 --img ./data/input_frames --output ./data/output_frames
```

---

## Interpolation Modes

This project offers two distinct interpolation modes: the standard direct approach and an advanced recursive method. It's important to understand their differences to choose the most suitable one for your needs and model version.

### 1\. Standard Interpolation (Recommended)

- **How to use:** Do **NOT** include the `--recursive` flag in your command. This is the default behavior.
- **Underlying Function:** `make_inference` (when `model.version >= 3.9`)
- **Mechanism:** This mode directly utilizes the `timestep` parameter of the RIFE model's `inference` function. For a given `--multi N`, it calculates `N-2` intermediate frames by evenly distributing the `timestep` values across the `0` to `1` interval (e.g., `1/(N-1)`, `2/(N-1)`, ..., `(N-2)/(N-1)`).
- **When to use:** This is the **recommended mode** for most modern RIFE models (especially versions 3.9 and higher, like `rifev4_25`). These models are typically trained to perform robustly across a continuous range of `timestep` values, providing the most flexible and consistent results for any `N` specified by `--multi`.

### 2\. Recursive Interpolation (Advanced / Experimental)

- **How to use:** Include the `--recursive` flag in your command (e.g., `python inference_img.py --recursive ...`).
- **Underlying Function:** `make_inference_recursive` (for `model.version >= 3.9`)
- **Mechanism:** This mode implements a more complex, multi-stage iterative refinement strategy. Instead of a single direct inference per intermediate frame, it performs several passes, incrementally refining flow and mask predictions for a sequence of frames. It's designed to build up the interpolated sequence through a series of specialized inferences.
- **When to use:** This mode might be considered for:
  - **Exploring alternative strategies:** If you are conducting research or experiments into different methods of flow-based interpolation refinement.
  - **Specific model behaviors:** In rare cases, some RIFE versions or custom-trained models might exhibit unique characteristics where iterative refinement yields marginally better results for specific content, although this is less common with general-purpose modern RIFE models.
- **Important Considerations:**
  - **Complexity:** The internal logic of `make_inference_recursive` is intricate.
  - **Performance/Quality:** For general usage with modern RIFE models (v3.9+), the **Standard Interpolation** mode is usually more efficient and delivers comparable or superior visual quality because the core `model.inference` is already highly optimized for direct `timestep` interpolation. The recursive mode might not offer significant advantages and could potentially introduce unique artifacts or increase processing time.
  - **`--multi` Compatibility:** While the code aims for flexibility, recursive bisection-like algorithms inherently work most efficiently and predictably when the number of _inserted_ frames (`--multi - 2`) plus one (i.e., `--multi - 1`) is a power of 2.

**For optimal results and ease of use, we strongly recommend starting with the Standard Interpolation mode (without the `--recursive` flag), as it aligns directly with the capabilities of most modern RIFE models.**

---

## 📈 Performance & Quality

- **Model Version:** The RIFE `4.25` model has shown good suitability for post-processing videos, including those generated by diffusion models, providing smooth and visually appealing transitions.
- **`--scale`:** For high-resolution inputs (e.g., 4K), using `--scale 0.5` or `--scale 0.25` is advised to manage VRAM usage and potentially improve speed, with a slight trade-off in fine detail.
- **Threads & Queues:** The multi-threaded I/O pipeline (`_thread.start_new_thread(build_read_buffer, ...)` and `_thread.start_new_thread(clear_write_buffer, ...)`) is crucial for maintaining a high processing throughput and preventing common out-of-memory issues, especially when dealing with large numbers of frames.

## 🤝 Contribution

Contributions are welcome\! If you have suggestions for improvements, new features, or bug fixes, please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
