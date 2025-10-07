# 파라미터(문서 기준)

> **원본 근거**: 첨부된 기술 문서(`study-asc-iot-ai-20230323-2050.docx`)를 소스 오브 트루스로 삼습니다.

이 레포지토리의 기본값은 `config/ears_config.json`에 반영되어 있으며, 주요 항목은 다음과 같습니다.

- Audio: sample_rate=16000, mono, duration=30s
- Framing: frame_length=16896, hop_length=512
- Features(Log-Mel): n_mels=30, n_fft=1024, hop_length=512, center=False
- Model: 30×32 입력, 간단 CNN(Conv→Pool×2 → Dense)
- Training: epochs=30, batch_size=500, optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True)

**문서의 값과 다를 경우** `config/ears_config.json`을 수정하고 아래 스크립트의 `--config` 인자를 활용하세요.
