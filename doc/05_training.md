# 학습

- 모델 구조(간단 CNN):
  - Conv2D(16, 3x3) → MaxPool(2x2) → Conv2D(16, 3x3) → MaxPool(2x2) → Flatten → Dense(9) → Dense(3, softmax)
- Optimizer: `SGD(lr=0.01, momentum=0.9, nesterov=True)`
- Loss: `categorical_crossentropy`
- Epochs: 30 (예시), Batch size: 500

출력:
- `Output/model.h5`
- 학습 곡선(loss) 이미지 `Output/train_val_loss.png`
