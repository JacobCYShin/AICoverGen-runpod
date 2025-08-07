#!/usr/bin/env python3
"""
RunPod Serverless AICoverGen API 테스트 코드
Colab에서 사용 가능한 테스트 스니펫
"""

import requests
import base64
import json
import time
import os

# ====== 사용자 설정 ======
API_KEY = ""  # RunPod API 키 입력
ENDPOINT_ID = ""  # RunPod Endpoint ID 입력

# ====== 테스트 설정 ======
VOICE_AUDIO_PATH = "tmp/Vocals_No_Noise.wav"  # 보컬 오디오 파일 경로
INSTRUMENT_AUDIO_PATH = "tmp/Instrumental.wav"  # 악기 오디오 파일 경로
VOICE_MODEL = "Jungkook"  # 사용할 보이스 모델명
PITCH_ADJUST = 0  # 피치 조정 (-12 ~ +12)
OUTPUT_FORMAT = "mp3"  # 출력 포맷 (mp3, wav)

# ====== 함수 정의 ======
def encode_audio_to_base64(file_path):
    """오디오 파일을 base64로 인코딩"""
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

def decode_base64_to_audio(base64_string, output_path):
    """base64를 오디오 파일로 디코딩"""
    audio_data = base64.b64decode(base64_string)
    with open(output_path, "wb") as f:
        f.write(audio_data)
    return output_path

def test_health_check():
    """헬스체크 테스트"""
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "operation": "health_check"
        }
    }

    print("🏥 헬스체크 테스트 중...")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 서버 정상 작동")
            print(f"   - 디바이스: {result.get('device', 'N/A')}")
            print(f"   - GPU 사용 가능: {result.get('gpu_available', 'N/A')}")
            print(f"   - 사용 가능한 모델: {result.get('available_models', [])}")
            return True
        else:
            print(f"❌ 헬스체크 실패: {response.status_code}")
            print(f"응답: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 헬스체크 중 오류: {str(e)}")
        return False

def test_list_models():
    """사용 가능한 모델 목록 테스트"""
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "operation": "list_models"
        }
    }

    print("📋 사용 가능한 모델 목록 조회 중...")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            models = result.get('models', [])
            count = result.get('count', 0)
            print(f"✅ 모델 목록 조회 완료")
            print(f"   - 총 모델 수: {count}")
            print(f"   - 사용 가능한 모델: {models}")
            return models
        else:
            print(f"❌ 모델 목록 조회 실패: {response.status_code}")
            print(f"응답: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ 모델 목록 조회 중 오류: {str(e)}")
        return []

def test_generate_cover():
    """AI 커버 생성 테스트"""

    # 1. 입력 오디오 → base64 인코딩
    print("🎵 오디오 파일 인코딩 중...")
    try:
        voice_audio = encode_audio_to_base64(VOICE_AUDIO_PATH)
        instrument_audio = encode_audio_to_base64(INSTRUMENT_AUDIO_PATH)
        print(f"✅ 인코딩 완료")
        print(f"   - 보컬 파일 크기: {len(voice_audio)} chars")
        print(f"   - 악기 파일 크기: {len(instrument_audio)} chars")
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        return

    # 2. API 요청 구성
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "operation": "generate_cover_from_separate_audio",
            "params": {
                "voice_audio": voice_audio,
                "instrument_audio": instrument_audio,
                "voice_model": VOICE_MODEL,
                "pitch_adjust": PITCH_ADJUST,
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
                "output_format": OUTPUT_FORMAT
            }
        }
    }

    # 3. 요청 전송
    print("🚀 AI 커버 생성 요청 전송 중...")
    print(f"📊 요청 정보:")
    print(f"   - 보이스 모델: {VOICE_MODEL}")
    print(f"   - 피치 조정: {PITCH_ADJUST}")
    print(f"   - 출력 포맷: {OUTPUT_FORMAT}")

    start_time = time.time()

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=300)  # 5분 타임아웃

        if response.status_code != 200:
            print(f"❌ 요청 실패: {response.status_code}")
            print(f"응답: {response.text}")
            return

    except requests.exceptions.Timeout:
        print("❌ 요청 타임아웃 (5분 초과)")
        return
    except Exception as e:
        print(f"❌ 요청 중 오류: {str(e)}")
        return

    processing_time = time.time() - start_time
    print(f"⏱️ 처리 시간: {processing_time:.2f}초")

    # 4. 응답 결과 처리
    try:
        result = response.json()
        print("✅ 응답 수신 완료")

        if "error" in result:
            print(f"❌ 서버 에러: {result['error']}")
            if "traceback" in result:
                print(f"상세 에러: {result['traceback']}")
            return

        # 출력 오디오 추출
        if "output_audio" in result:
            output_base64 = result["output_audio"]

            # 5. 디코딩 및 저장
            output_filename = f"ai_cover_{VOICE_MODEL}.{OUTPUT_FORMAT}"
            decode_base64_to_audio(output_base64, output_filename)

            print(f"🎉 결과 저장 완료 → {output_filename}")
            print(f"📁 파일 경로: {os.path.abspath(output_filename)}")
            print(f"📊 파일 크기: {os.path.getsize(output_filename)} bytes")

            # 추가 정보 출력
            if "parameters" in result:
                params = result["parameters"]
                print(f"\n📋 처리 정보:")
                for key, value in params.items():
                    print(f"   - {key}: {value}")

            print(f"📊 서버 응답 파일 크기: {result.get('size', 'N/A')} bytes")

        else:
            print("❌ 예상된 출력 형식이 아닙니다.")
            print(f"응답 구조: {json.dumps(result, indent=2, ensure_ascii=False)}")

    except json.JSONDecodeError:
        print("❌ JSON 디코딩 실패")
        print(f"응답 텍스트: {response.text}")

# ====== 고급 테스트 함수 ======
def test_multiple_models():
    """여러 모델로 테스트"""
    models = ["Jungkook"]  # 사용 가능한 모델들

    for model in models:
        print(f"\n🎤 {model} 모델 테스트 중...")
        global VOICE_MODEL
        VOICE_MODEL = model
        test_generate_cover()
        time.sleep(3)  # 요청 간 간격

def test_pitch_variations():
    """다양한 피치로 테스트"""
    pitches = [-2, 0, 2]  # 낮음, 원본, 높음

    for pitch in pitches:
        print(f"\n🎵 피치 {pitch:+d} 테스트 중...")
        global PITCH_ADJUST
        PITCH_ADJUST = pitch
        test_generate_cover()
        time.sleep(3)

def test_different_formats():
    """다양한 출력 포맷 테스트"""
    formats = ["mp3", "wav"]

    for fmt in formats:
        print(f"\n🎵 {fmt.upper()} 포맷 테스트 중...")
        global OUTPUT_FORMAT
        OUTPUT_FORMAT = fmt
        test_generate_cover()
        time.sleep(3)

# ====== 실행 ======
if __name__ == "__main__":
    print("🎵 RunPod Serverless AICoverGen API 테스트 시작")
    print("="*60)

    # 1. 헬스체크
    if not test_health_check():
        print("❌ 서버가 정상 작동하지 않습니다. 종료합니다.")
        exit(1)

    print("\n" + "="*60)

    # 2. 사용 가능한 모델 확인
    available_models = test_list_models()
    if VOICE_MODEL not in available_models:
        print(f"⚠️ 경고: '{VOICE_MODEL}' 모델이 사용 가능한 모델 목록에 없습니다.")
        if available_models:
            print(f"사용 가능한 모델 중 하나를 선택하세요: {available_models}")
            VOICE_MODEL = available_models[0]
            print(f"자동으로 '{VOICE_MODEL}' 모델을 사용합니다.")

    print("\n" + "="*60)

    # 3. 기본 테스트
    test_generate_cover()

    # 추가 테스트 (주석 해제하여 사용)
    # test_multiple_models()
    # test_pitch_variations()
    # test_different_formats()

    print("\n🎉 모든 테스트 완료!")
