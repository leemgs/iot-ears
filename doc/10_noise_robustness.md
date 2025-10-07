
# 노이즈 강건성 평가 (SNR)

학습된 모델의 잡음 환경 성능을 SNR(dB)별로 측정합니다.

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

출력:
- `OutputNoise/snr_results.json` : 각 SNR별 정확도 + Confusion Matrix
- `OutputNoise/snr_results.csv`  : SNR vs Accuracy 표
