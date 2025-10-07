# 환경 준비

- Python 3.7 (문서 기준), conda 권장
- 라이브러리 버전
  - TensorFlow **1.14.0**
  - Keras **2.2.4**
  - librosa **0.9.2**
  - h5py **2.10.0**
- 설치: `pip install -r requirements.txt` 또는 `scripts/create_conda_env.sh` 실행

주의: TF1.x와 Keras 2.2.4 조합을 사용합니다. TF2.x에서는 `tf.compat.v1.disable_eager_execution()`이 필요합니다.
