# 데이터 준비

`examples/Dataset/TrainSet.txt` 포맷:
```
bus.wav bus
park.wav park
home.wav home
```

세부 라벨 → 3대 클래스 매핑:
- indoor: cafeRestaurant, home, office, shoppingCenter
- outdoor: cityCenter, park, residentialArea
- vehicle: bus, car, subway, train, tramway

오디오는 로딩시 **16 kHz, mono, duration 30 s**, dtype float32로 통일합니다.
STM32 설정과 **샘플레이트/채널**이 일치해야 합니다.
