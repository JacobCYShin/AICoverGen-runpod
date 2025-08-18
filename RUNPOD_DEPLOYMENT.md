# RunPod Serverless 배포 가이드

## 1. 사전 준비

### RunPod CLI 설치
```bash
pip install runpod
```

### RunPod 로그인
```bash
runpod login
```

## 2. 모델 파일 준비

RunPod Serverless에서 사용할 모델 파일들을 준비해야 합니다:

### 필수 모델 파일들
- `rvc_models/hubert_base.pt` - HuBERT 모델
- `rvc_models/rmvpe.pt` - RMVPE 모델  
- `rvc_models/Jimin/` - Jimin 보컬 모델 (기타 모델들도 필요시)

### 모델 파일 업로드
```bash
# 방법 1: 자동 업로드 스크립트 사용
python3 upload_models_to_runpod.py

# 방법 2: 수동 업로드
runpod upload rvc_models/ /runpod-volume/rvc_models
runpod upload mdxnet_models/ /runpod-volume/mdxnet_models
```

## 3. 서버리스 배포

### 배포 명령어
```bash
runpod deploy --config runpod.yaml
```

### 배포 후 확인
```bash
# 배포된 엔드포인트 확인
runpod list

# 헬스체크 테스트
curl -X POST https://your-endpoint.runpod.net/health_check
```

## 4. API 사용법

### 헬스체크
```bash
curl -X POST https://your-endpoint.runpod.net/health_check
```

### 사용 가능한 모델 목록
```bash
curl -X POST https://your-endpoint.runpod.net/list_models
```

### 커버 생성
```bash
curl -X POST https://your-endpoint.runpod.net/generate_cover_from_separate_audio \
  -H "Content-Type: application/json" \
  -d '{
    "voice_audio": "base64_encoded_voice_audio",
    "instrument_audio": "base64_encoded_instrument_audio", 
    "voice_model": "Jimin",
    "pitch_adjust": 0,
    "index_rate": 0.5,
    "filter_radius": 3,
    "rms_mix_rate": 0.25,
    "protect": 0.33,
    "f0_method": "rmvpe",
    "reverb_rm_size": 0.25,
    "reverb_wet": 0.4,
    "reverb_dry": 0.6,
    "reverb_damping": 0.5,
    "main_gain": 0,
    "backup_gain": 0,
    "inst_gain": 0,
    "output_format": "mp3"
  }'
```

## 5. 최적화 사항

### 메모리 관리
- GPU 메모리 자동 정리 (`torch.cuda.empty_cache()`)
- 가비지 컬렉션 자동 실행
- CUDA 메모리 할당 최적화

### 성능 최적화
- 22kHz 샘플레이트로 빠른 처리
- 간단한 믹싱 알고리즘 사용
- 불필요한 오디오 효과 제거

### 서버리스 특화
- 임시 디렉토리 사용으로 디스크 공간 절약
- 에러 처리 및 메모리 정리 강화
- 비동기 처리 지원

## 6. 모니터링

### 로그 확인
```bash
runpod logs <pod_id>
```

### 리소스 사용량 확인
```bash
runpod stats <pod_id>
```

## 7. 문제 해결

### 일반적인 문제들
1. **GPU 메모리 부족**: 모델 크기 줄이기 또는 배치 크기 조정
2. **타임아웃**: 오디오 길이 제한 또는 처리 시간 최적화
3. **모델 로딩 실패**: 모델 파일 경로 확인

### 디버깅
```bash
# 상세 로그 확인
runpod logs <pod_id> --follow

# 컨테이너 내부 접속 (필요시)
runpod exec <pod_id> bash
```

## 8. 비용 최적화

### 권장 설정
- **GPU**: RTX 4090 또는 A100 (빠른 처리)
- **메모리**: 16GB (안정적 처리)
- **CPU**: 4코어 (충분한 처리 능력)
- **디스크**: 20GB (모델 파일 저장)

### 사용량 모니터링
- RunPod 대시보드에서 실시간 사용량 확인
- 비용 알림 설정
- 사용하지 않는 시간에는 자동 스케일링
