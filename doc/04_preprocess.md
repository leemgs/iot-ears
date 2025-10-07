# 전처리

- 프레임 길이 `frame_length=16896`, `hop_length=512`
- 각 프레임에서 **MelSpectrogram** 계산: `n_mels=30`, `n_fft=1024`, `hop_length=512`, `center=False`
- dB 스케일로 **Log‑Mel** 변환 후, **StandardScaler()**로 평균 0/분산 1 표준화
- 입력 텐서는 (30, 32, 1)

산출물(`Output/`):
- `x_train.csv`, `y_train.csv`, `x_val.csv`, `y_val.csv`, `x_test.csv`, `y_test.csv`
