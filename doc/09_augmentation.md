
# 데이터 증강 (Augmentation)

학습 일반화를 위해 파형 단계에서 다음 증강을 제공합니다.

- **White Noise (SNR)**: `--snr_list 20,10,0` 등
- **Time Shift**: `--shift_ratio 0.1` (신호 길이의 비율내 무작위 이동)
- **Pitch Shift**: `--pitch_steps -1,1`
- **Time Stretch**: `--time_stretch 0.9,1.1`

사용 예시:
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
```
생성물은 `OutputAug/`에 저장되며, `scaler.npz`도 같이 저장되어 **노이즈 평가**에 재사용됩니다.
