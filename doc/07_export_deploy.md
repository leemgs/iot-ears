# TFLite/STM32 배포

1) `export_tflite.py`로 **INT8** 양자화된 `model.tflite` 생성
2) **STM32Cube.AI** 툴에서 `model.h5` 또는 `model.tflite` 로드 → C 코드 생성
3) STM32CubeIDE에서 빌드/플래시

빌드 팁:
- 링커 스크립트 `Heap/Stack` 최소값 조정
- `-Os` 최적화
- 불필요 섹션 제거(`Discard unused section`)
- float ABI 설정 확인

버그픽스 포인트(문서 참조):
- `ASC_NN_Run()` 처리 경로 및 배치 크기 매크로 확인
