
# EARS – Edge Acoustic Recognition System (ASC for IoT)

A reproducible reference implementation of **Acoustic Scene Classification (ASC)** designed for **STM32-based IoT devices**.

This repository provides:

* A complete **Python pipeline** for **preprocessing → training → evaluation** of a 3-class CNN (indoor / outdoor / vehicle)
* Scripts to **export Keras (`.h5`) → TensorFlow Lite (`.tflite`)** with **INT8 post-training quantization**
* Structured documentation in `./doc/` and detailed STM32 deployment notes in `./stm32/`
* Licensed under **Apache-2.0**

> This codebase is organized based on the attached Korean technical document, which specifies the dataset, feature extraction methods, model design, and STM32 deployment workflow.

---

## 🚀 Quickstart

### 0) Environment Setup

```bash
# (Recommended) Create a conda environment pinned to the required versions
bash scripts/create_conda_env.sh

# Or install dependencies manually
pip install -r requirements.txt
```

### 1) Prepare the Dataset

* Place your audio files under `examples/Dataset/`
* Create a metadata file `TrainSet.txt` with lines in the format:
  `<path> <label>`
  Example:

  ```
  home.wav home
  ```
* Fine-grained labels defined in the documentation are mapped to `{indoor, outdoor, vehicle}`. See `doc/03_data.md` for details.

### 2) Preprocess Audio (Log-Mel, Standardization, Split)

```bash
python -m src.python.preprocess \
  --dataset_dir examples/Dataset \
  --meta TrainSet.txt \
  --out_dir Output
```

### 3) Train the CNN Model

```bash
python -m src.python.train \
  --in_dir Output \
  --epochs 30 \
  --batch_size 500 \
  --lr 0.01 \
  --momentum 0.9
```

### 4) Evaluate the Model & Generate Confusion Matrix

```bash
python -m src.python.evaluate --in_dir Output
```

### 5) Export to TensorFlow Lite (INT8 Quantized)

```bash
python -m src.python.export_tflite --in_dir Output
```

### 6) (Manual) Convert to STM32 C Code

Use **STM32Cube.AI** for model conversion.
Refer to `doc/07_export_deploy.md` and `stm32/README.md` for deployment checklists, including tips for:

* Heap/stack configuration
* `-Os` compiler optimization
* Floating-point ABI settings

---

## 📁 Repository Structure

```
src/python/
  preprocess.py         # Framing → Log-Mel → Standardize → Split → Save CSV
  train.py              # Keras 2.2.4 + TF 1.14 training, saves model.h5
  evaluate.py           # Test set accuracy + confusion matrix visualization
  export_tflite.py      # INT8 post-training quantization
  utils/audio.py        # Audio processing helpers (Librosa)
  utils/io.py           # CSV/NPY/Path handling helpers

doc/                    # Step-by-step documentation (Korean)
stm32/                  # STM32Cube.AI deployment notes
examples/Dataset/       # Place your .wav files here + TrainSet.txt
scripts/                # Environment setup scripts
```

---

## 📜 License

Licensed under **Apache-2.0** (see `LICENSE.md`).

---

## ⚙️ Central Configuration

All official parameters are defined in `config/ears_config.json`.
Scripts accept a `--config` argument to override defaults easily:

```bash
python -m src.python.preprocess \
  --dataset_dir examples/Dataset \
  --meta TrainSet.txt \
  --out_dir Output \
  --config config/ears_config.json

python -m src.python.train \
  --in_dir Output \
  --config config/ears_config.json
```

---

## 🧰 Dataset Helper

If your dataset is organized as `<dataset_dir>/<fine_label>/*.wav`,
you can auto-generate `TrainSet.txt` with:

```bash
python -m src.python.tools.make_trainset \
  --dataset_dir examples/Dataset \
  --out examples/Dataset/TrainSet.txt
```

---

## 🔁 Data Augmentation Pipeline

Expand the training set with waveform-level augmentation:

```bash
python -m src.python.preprocess_aug \
  --dataset_dir examples/Dataset \
  --meta TrainSet.txt \
  --out_dir OutputAug \
  --snr_list 20,10,0 \
  --pitch_steps -1,1 \
  --time_stretch 0.9,1.1 \
  --shift_ratio 0.1 \
  --config config/ears_config.json

python -m src.python.train \
  --in_dir OutputAug \
  --config config/ears_config.json
```

---

## 🔊 Noise Robustness Evaluation

Test the trained model’s robustness under various SNR conditions:

```bash
python -m src.python.eval_noise \
  --dataset_dir examples/Dataset \
  --meta TrainSet.txt \
  --model Output/model.h5 \
  --scaler Output/scaler.npz \
  --snr_list clean,20,10,0 \
  --out_dir OutputNoise \
  --config config/ears_config.json
```

---

### 📌 Notes

* The pipeline is optimized for embedded deployment, especially STM32 microcontrollers.
* For best results, review the hardware-specific optimization checklist before deployment.
* All steps are reproducible and can be automated through `Makefile` or shell scripts for CI/CD integration.

---
