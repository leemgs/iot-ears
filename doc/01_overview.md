# 프로젝트 개요

이 레포지토리는 IoT 보드(예: **STM32 B-L475E-IOT01A**)의 마이크로 수집한 환경음을 **실내 / 실외 / 차량내** 3개 클래스로 분류하는
Acoustic Scene Classification(ASC) 파이프라인의 재현용 소스입니다.

파이프라인: 수집 → 프레이밍 → **Log‑Mel Spectrogram** → 표준화 → 학습(CNN) → 평가 → **TFLite INT8** 변환 → STM32 C 코드 생성.

원문 문서의 상세 절차와 파라미터를 그대로 반영했습니다.
