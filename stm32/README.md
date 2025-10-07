# STM32Cube.AI 배포 체크리스트

1. `Output/model.h5` 또는 `Output/model.tflite`를 STM32Cube.AI에 로드하여 C 코드 생성
2. 생성된 `asc.*`, `asc_data.*` 파일을 프로젝트에 반영
3. 링커 스크립트 Heap/Stack 최소값 조정 (RAM 부족 오류 방지)
4. 컴파일 플래그
   - `-Os` 최적화
   - 디버그 레벨 None
   - Linker: Discard unused sections
   - (필요시) float ABI 옵션 확인
5. 배치 크기 매크로 및 `ASC_NN_Run()` 코드 경로 확인
6. UART 로깅: `SENSING1_USE_PRINTF` 매크로 활성화로 시리얼 로그 확인
