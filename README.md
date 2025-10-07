# EARS – Edge Acoustic Recognition System (ASC for IoT)

Reproducible reference implementation for **Acoustic Scene Classification (ASC)** targeting STM32-based IoT devices.
This repository includes:
- Python pipeline to **preprocess → train → evaluate** a 3-class CNN (indoor/outdoor/vehicle)
- Scripts to **export Keras (.h5) → TensorFlow Lite (.tflite)** with INT8 post-training quantization
- Structured docs in `./doc/` and STM32 deployment notes in `./stm32/`
- Apache-2.0 licensed

> This codebase is organized from the attached technical document (Korean) that specifies data, features, model, and STM32 deployment steps.

## Quickstart

### 0) Environment
```bash
# (Recommended) Create a conda env pinned to the specified versions
bash scripts/create_conda_env.sh
# Or install manually
pip install -r requirements.txt
```

### 1) Prepare dataset
- Place audio files in `examples/Dataset/` and create `TrainSet.txt` with lines: `<path> <label>` (e.g., `home.wav home`).
- Supported fine-grained labels in the doc are mapped to {indoor,outdoor,vehicle}. See `doc/03_data.md`.

### 2) Preprocess (Log-Mel, standardize, split)
```bash
python -m src.python.preprocess   --dataset_dir examples/Dataset   --meta TrainSet.txt   --out_dir Output
```

### 3) Train CNN
```bash
python -m src.python.train   --in_dir Output   --epochs 30   --batch_size 500   --lr 0.01   --momentum 0.9
```

### 4) Evaluate & Confusion Matrix
```bash
python -m src.python.evaluate --in_dir Output
```

### 5) Export to TFLite (INT8)
```bash
python -m src.python.export_tflite --in_dir Output
```

### 6) (Manual) Convert to STM32 C (with STM32Cube.AI)
See `doc/07_export_deploy.md` and `stm32/README.md` for checklists and typical gotchas (heap/stack, -Os, float ABI, etc.).

## Repository Layout
```
src/python/
  preprocess.py         # framing → Log-Mel → standardize → split → save CSV
  train.py              # Keras 2.2.4 + TF 1.14 training, saves model.h5
  evaluate.py           # test set accuracy + confusion matrix figure
  export_tflite.py      # INT8 post-training quantization
  utils/audio.py        # audio helpers (Librosa)
  utils/io.py           # IO helpers for CSV/NPY/paths

doc/                    # step-by-step documentation (Korean)
stm32/                  # STM32Cube.AI deployment notes
examples/Dataset/       # put wav files here and a TrainSet.txt
scripts/                # env setup
```

## License
Apache-2.0 (see `LICENSE.md`)


## Central Config
공식 문서의 파라미터는 `config/ears_config.json`에 정리되어 있습니다. 스크립트는 `--config config/ears_config.json`로 쉽게 덮어쓸 수 있습니다.

예)
```bash
python -m src.python.preprocess --dataset_dir examples/Dataset --meta TrainSet.txt --out_dir Output --config config/ears_config.json
python -m src.python.train --in_dir Output --config config/ears_config.json
```


## Dataset Helper
폴더 구조가 `<dataset_dir>/<fine_label>/*.wav` 인 경우, 다음으로 `TrainSet.txt`를 자동 생성할 수 있습니다.
```bash
python -m src.python.tools.make_trainset --dataset_dir examples/Dataset --out examples/Dataset/TrainSet.txt
```

## Augmentation Pipeline
파형 단계에서 증강을 적용하여 학습 세트를 확장합니다.
```bash
python -m src.python.preprocess_aug --dataset_dir examples/Dataset --meta TrainSet.txt --out_dir OutputAug --snr_list 20,10,0 --pitch_steps -1,1 --time_stretch 0.9,1.1 --shift_ratio 0.1 --config config/ears_config.json
python -m src.python.train --in_dir OutputAug --config config/ears_config.json
```

## Noise Robustness Evaluation
학습된 모델의 SNR 강건성을 평가합니다.
```bash
python -m src.python.eval_noise --dataset_dir examples/Dataset --meta TrainSet.txt --model Output/model.h5 --scaler Output/scaler.npz --snr_list clean,20,10,0 --out_dir OutputNoise --config config/ears_config.json
```
