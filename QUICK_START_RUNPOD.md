# RunPod Serverless AICoverGen 빠른 시작 가이드

## 1. 사전 준비

### RunPod CLI 설치 및 로그인
```bash
pip install runpod
runpod login
```

### 필수 모델 파일 확인
```
rvc_models/
├── hubert_base.pt
├── rmvpe.pt
└── Jimin/
    ├── Jimin.pth
    └── Jimin.index
```

## 2. 모델 업로드

### 자동 업로드 (권장)
```bash
python3 upload_models_to_runpod.py
```

### 수동 업로드
```bash
runpod upload rvc_models/ /runpod-volume/rvc_models
runpod upload mdxnet_models/ /runpod-volume/mdxnet_models
```

## 3. 서버리스 배포

```bash
runpod deploy --config runpod.yaml
```

배포 후 Endpoint ID를 확인하세요:
```bash
runpod list
```

## 4. API 테스트

### 테스트 코드 설정
`test_runpod_aicovergen.py` 파일에서 다음을 수정하세요:

```python
API_KEY = "YOUR_RUNPOD_API_KEY"  # RunPod API 키
ENDPOINT_ID = "YOUR_ENDPOINT_ID"  # 배포된 Endpoint ID
```

### 테스트 실행
```bash
# 방법 1: 직접 실행
python3 test_runpod_aicovergen.py

# 방법 2: 간단한 실행 스크립트 사용
python3 run_test.py
```

## 5. API 사용법

### 헬스체크
```bash
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync \
  -H "Authorization: Bearer {API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"input": {"operation": "health_check"}}'
```

### 모델 목록 조회
```bash
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync \
  -H "Authorization: Bearer {API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"input": {"operation": "list_models"}}'
```

### AI 커버 생성
```bash
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync \
  -H "Authorization: Bearer {API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "operation": "generate_cover_from_separate_audio",
      "params": {
        "voice_audio": "base64_encoded_voice_audio",
        "instrument_audio": "base64_encoded_instrument_audio",
        "voice_model": "Jimin",
        "pitch_adjust": 0,
        "output_format": "mp3"
      }
    }
  }'
```

## 6. 주요 파라미터

### 필수 파라미터
- `voice_audio`: 보컬 오디오 (base64)
- `instrument_audio`: 악기 오디오 (base64)
- `voice_model`: 사용할 보이스 모델명

### 선택 파라미터
- `pitch_adjust`: 피치 조정 (-12 ~ +12)
- `index_rate`: 인덱스 비율 (0.0 ~ 1.0)
- `filter_radius`: 필터 반지름 (0 ~ 7)
- `rms_mix_rate`: RMS 믹스 비율 (0.0 ~ 1.0)
- `protect`: 보호 값 (0.0 ~ 1.0)
- `f0_method`: F0 방법 ("rmvpe", "pm", "harvest", "crepe", "rmvpe_gpu")
- `reverb_rm_size`: 리버브 공간 크기 (0.0 ~ 1.0)
- `reverb_wet`: 리버브 웻 레벨 (0.0 ~ 1.0)
- `reverb_dry`: 리버브 드라이 레벨 (0.0 ~ 1.0)
- `reverb_damping`: 리버브 댐핑 (0.0 ~ 1.0)
- `main_gain`: 메인 보컬 게인 (dB)
- `backup_gain`: 백업 보컬 게인 (dB)
- `inst_gain`: 악기 게인 (dB)
- `output_format`: 출력 포맷 ("mp3", "wav")

## 7. 응답 형식

### 성공 응답
```json
{
  "success": true,
  "output_audio": "base64_encoded_audio",
  "filename": "cover_Jimin.mp3",
  "size": 1234567,
  "model_used": "Jimin",
  "parameters": {
    "pitch_adjust": 0,
    "index_rate": 0.5,
    "output_format": "mp3"
  }
}
```

### 에러 응답
```json
{
  "error": "Error message",
  "traceback": "Detailed error traceback"
}
```

## 8. 문제 해결

### 일반적인 문제들
1. **모델 파일 없음**: `/runpod-volume/rvc_models/` 경로에 모델 파일들이 있는지 확인
2. **GPU 메모리 부족**: 더 큰 GPU 인스턴스 사용
3. **타임아웃**: 오디오 파일 크기 줄이기 또는 처리 시간 최적화

### 로그 확인
```bash
runpod logs {POD_ID}
```

### 컨테이너 재시작
```bash
runpod restart {POD_ID}
```

## 9. 비용 최적화

### 권장 설정
- **GPU**: RTX 4090 (빠른 처리)
- **메모리**: 16GB (안정적 처리)
- **CPU**: 4코어 (충분한 처리 능력)

### 사용량 모니터링
- RunPod 대시보드에서 실시간 사용량 확인
- 비용 알림 설정

## 10. 예제 코드

### Python 예제
```python
import requests
import base64

def generate_ai_cover(api_key, endpoint_id, voice_audio_path, instrument_audio_path, voice_model):
    # 오디오 파일 인코딩
    with open(voice_audio_path, "rb") as f:
        voice_audio = base64.b64encode(f.read()).decode("utf-8")
    
    with open(instrument_audio_path, "rb") as f:
        instrument_audio = base64.b64encode(f.read()).decode("utf-8")
    
    # API 요청
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "operation": "generate_cover_from_separate_audio",
            "params": {
                "voice_audio": voice_audio,
                "instrument_audio": instrument_audio,
                "voice_model": voice_model,
                "output_format": "mp3"
            }
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

# 사용 예제
result = generate_ai_cover(
    api_key="YOUR_API_KEY",
    endpoint_id="YOUR_ENDPOINT_ID",
    voice_audio_path="tmp/Vocals_No_Noise.wav",
    instrument_audio_path="tmp/Instrumental.wav",
    voice_model="Jimin"
)

if "output_audio" in result:
    # 결과 오디오 저장
    audio_data = base64.b64decode(result["output_audio"])
    with open("output.mp3", "wb") as f:
        f.write(audio_data)
    print("AI 커버 생성 완료!")
else:
    print(f"에러: {result.get('error', 'Unknown error')}")
```
