# AICoverGen RunPod API

이 문서는 RunPod에서 AICoverGen을 실행하기 위한 API 사용법을 설명합니다.

## API 엔드포인트

### 1. Health Check
시스템 상태와 사용 가능한 모델을 확인합니다.

**요청:**
```json
{
  "operation": "health_check"
}
```

**응답:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "available_models": ["model1", "model2", ...],
  "gpu_available": true
}
```

### 2. List Models
사용 가능한 음성 모델 목록을 반환합니다.

**요청:**
```json
{
  "operation": "list_models"
}
```

**응답:**
```json
{
  "models": ["model1", "model2", ...],
  "count": 5
}
```

### 3. Generate Cover from Separate Audio
분리된 보컬과 악기 오디오 파일로부터 AI 커버를 생성합니다.

**요청:**
```json
{
  "operation": "generate_cover_from_separate_audio",
  "params": {
    "voice_audio": "base64_encoded_voice_audio",
    "instrument_audio": "base64_encoded_instrument_audio",
    "voice_model": "model_name",
    "pitch_adjust": 0,
    "index_rate": 0.5,
    "filter_radius": 3,
    "rms_mix_rate": 0.25,
    "protect": 0.33,
    "f0_method": "rmvpe",
    "crepe_hop_length": 128,
    "pitch_change_all": 0,
    "reverb_rm_size": 0.15,
    "reverb_wet": 0.2,
    "reverb_dry": 0.8,
    "reverb_damping": 0.7,
    "main_gain": 0,
    "backup_gain": 0,
    "inst_gain": 0,
    "output_format": "mp3"
  }
}
```

**응답:**
```json
{
  "success": true,
  "output_audio": "base64_encoded_output_audio",
  "filename": "cover_model_name.mp3",
  "size": 1234567,
  "model_used": "model_name",
  "parameters": {
    "pitch_adjust": 0,
    "index_rate": 0.5,
    "filter_radius": 3,
    "rms_mix_rate": 0.25,
    "protect": 0.33,
    "f0_method": "rmvpe",
    "pitch_change_all": 0,
    "reverb_rm_size": 0.15,
    "reverb_wet": 0.2,
    "reverb_dry": 0.8,
    "reverb_damping": 0.7,
    "main_gain": 0,
    "backup_gain": 0,
    "inst_gain": 0,
    "output_format": "mp3"
  }
}
```

## 파라미터 설명

### 필수 파라미터
- `voice_audio`: Base64로 인코딩된 보컬 오디오 파일
- `instrument_audio`: Base64로 인코딩된 악기 오디오 파일
- `voice_model`: 사용할 음성 모델 이름

### 선택적 파라미터
- `pitch_adjust`: 보컬 변환용 피치 조정 (-12 ~ 12)
- `index_rate`: 인덱스 비율 (0.0 ~ 1.0)
- `filter_radius`: 필터 반지름 (0 ~ 7)
- `rms_mix_rate`: RMS 믹스 비율 (0.0 ~ 1.0)
- `protect`: 보호 값 (0.0 ~ 0.5)
- `f0_method`: F0 방법 ("rmvpe", "crepe", "harvest")
- `crepe_hop_length`: Crepe hop length (128)
- `pitch_change_all`: 전체 오디오 피치 변경 (-12 ~ 12)
- `reverb_rm_size`: 리버브 룸 사이즈 (0.0 ~ 1.0)
- `reverb_wet`: 리버브 웻 레벨 (0.0 ~ 1.0)
- `reverb_dry`: 리버브 드라이 레벨 (0.0 ~ 1.0)
- `reverb_damping`: 리버브 댐핑 (0.0 ~ 1.0)
- `main_gain`: AI 메인 보컬 볼륨 조정 (dB)
- `backup_gain`: 백업 보컬 볼륨 조정 (dB)
- `inst_gain`: 악기 볼륨 조정 (dB)
- `output_format`: 출력 형식 ("mp3", "wav")

## 사용 예제

### Python 클라이언트 예제

```python
import requests
import base64
import json

# 보컬과 악기 오디오 파일을 base64로 인코딩
with open("voice.wav", "rb") as f:
    voice_base64 = base64.b64encode(f.read()).decode('utf-8')

with open("instrument.wav", "rb") as f:
    instrument_base64 = base64.b64encode(f.read()).decode('utf-8')

# API 요청
payload = {
    "operation": "generate_cover_from_separate_audio",
    "params": {
        "voice_audio": voice_base64,
        "instrument_audio": instrument_base64,
        "voice_model": "your_model_name",
        "pitch_adjust": 0,
        "index_rate": 0.5,
        "f0_method": "rmvpe",
        "output_format": "mp3"
    }
}

# RunPod API 호출
response = requests.post("YOUR_RUNPOD_ENDPOINT", json=payload)
result = response.json()

if result.get("success"):
    # 출력 오디오 저장
    audio_data = base64.b64decode(result["output_audio"])
    with open(result["filename"], "wb") as f:
        f.write(audio_data)
    print("Cover generated successfully!")
else:
    print(f"Error: {result.get('error')}")
```

### cURL 예제

```bash
# Health check
curl -X POST "YOUR_RUNPOD_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{"operation": "health_check"}'

# Generate cover from separate audio
curl -X POST "YOUR_RUNPOD_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "generate_cover_from_separate_audio",
    "params": {
      "voice_audio": "base64_encoded_voice_audio",
      "instrument_audio": "base64_encoded_instrument_audio",
      "voice_model": "model_name",
      "pitch_adjust": 0,
      "index_rate": 0.5,
      "output_format": "mp3"
    }
  }'
```

## 처리 과정

1. **Voice Conversion**: RVC 모델을 사용하여 보컬 변환
2. **Audio Effects**: 리버브, 컴프레서 등 오디오 효과 적용
3. **Pitch Shifting**: 필요시 전체 오디오 피치 조정
4. **Audio Mixing**: AI 보컬, 백업 보컬, 악기를 믹싱
5. **Output**: 최종 결과물을 base64로 인코딩하여 반환

## 오류 처리

API는 오류 발생 시 다음과 같은 형식으로 응답합니다:

```json
{
  "error": "Error description"
}
```

일반적인 오류:
- `Voice model not found`: 존재하지 않는 모델명 사용
- `Invalid base64 audio data`: 잘못된 오디오 데이터
- `No model file exists`: 모델 파일이 존재하지 않음

## 모델 업로드

RunPod 환경에서 모델을 업로드하려면:
1. 모델 파일을 컨테이너의 `/workspace/rvc_models/` 디렉토리에 업로드
2. 모델은 `.pth` 파일과 `.index` 파일로 구성
3. 각 모델은 별도의 디렉토리에 저장

## 성능 최적화

- GPU 메모리 사용량을 고려하여 배치 크기 조정
- 긴 오디오 파일의 경우 청크 단위로 처리 고려
- 모델 캐싱을 위해 컨테이너 재시작 최소화
- 임시 파일은 자동으로 정리됨
