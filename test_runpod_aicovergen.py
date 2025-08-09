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
import argparse
import sys
import subprocess
import shutil
import tempfile

# ====== 커맨드 라인 인자 파싱 ======
def parse_arguments():
    """커맨드 라인 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="RunPod Serverless AICoverGen API 테스트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python test_runpod_aicovergen.py --api-key YOUR_API_KEY --endpoint-id YOUR_ENDPOINT_ID
  python test_runpod_aicovergen.py -k YOUR_API_KEY -e YOUR_ENDPOINT_ID
        """
    )
    
    parser.add_argument(
        "--api-key", "-k",
        required=True,
        help="RunPod API 키"
    )
    
    parser.add_argument(
        "--endpoint-id", "-e", 
        required=True,
        help="RunPod Endpoint ID"
    )
    
    parser.add_argument(
        "--voice-audio", "-v",
        default="tmp/Vocals_No_Noise.wav",
        help="보컬 오디오 파일 경로 (기본값: tmp/Vocals_No_Noise.wav)"
    )
    
    parser.add_argument(
        "--instrument-audio", "-i",
        default="tmp/Instrumental.wav", 
        help="악기 오디오 파일 경로 (기본값: tmp/Instrumental.wav)"
    )
    
    parser.add_argument(
        "--voice-model", "-m",
        default="Jungkook",
        help="사용할 보이스 모델명 (기본값: Jungkook)"
    )
    
    parser.add_argument(
        "--pitch-adjust", "-p",
        type=int,
        default=0,
        help="피치 조정 (-12 ~ +12, 기본값: 0)"
    )
    
    parser.add_argument(
        "--output-format", "-f",
        choices=["mp3", "wav"],
        default="mp3",
        help="출력 포맷 (기본값: mp3)"
    )
    
    parser.add_argument(
        "--test-mode",
        choices=["basic", "multiple-models", "pitch-variations", "different-formats"],
        default="basic",
        help="테스트 모드 (기본값: basic)"
    )
    
    parser.add_argument(
        "--trim-seconds",
        type=int,
        default=0,
        help="전송 전에 로컬에서 오디오 길이를 앞부분 N초로 트리밍하여 페이로드를 줄입니다. 0이면 비활성화"
    )
    
    return parser.parse_args()

# ====== 전역 변수 ======
# 커맨드 라인 인자에서 설정됨
API_KEY = None
ENDPOINT_ID = None
VOICE_AUDIO_PATH = None
INSTRUMENT_AUDIO_PATH = None
VOICE_MODEL = None
PITCH_ADJUST = None
OUTPUT_FORMAT = None
TRIM_SECONDS = 0

# ====== 유틸 ======
def unwrap_runpod_output(result: dict) -> dict:
    """RunPod runsync 응답에서 output 키가 있으면 언래핑합니다."""
    if isinstance(result, dict) and "output" in result and isinstance(result["output"], dict):
        return result["output"]
    return result


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


def maybe_trim_audio(input_path: str, trim_seconds: int) -> str:
    """ffmpeg가 있으면 앞부분 N초만 잘라 임시 파일로 저장해 반환합니다. 실패 시 원본 경로 반환."""
    if trim_seconds <= 0:
        return input_path
    if not shutil.which("ffmpeg"):
        print("⚠️ ffmpeg가 없어 트리밍을 건너뜁니다.")
        return input_path
    try:
        tmp_dir = tempfile.mkdtemp(prefix="aicovergen_trim_")
        output_path = os.path.join(tmp_dir, os.path.splitext(os.path.basename(input_path))[0] + f"_trim{trim_seconds}.wav")
        cmd = [
            "ffmpeg", "-y",
            "-t", str(trim_seconds),
            "-i", input_path,
            "-ac", "2",
            "-ar", "44100",
            output_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✂️ 트리밍 완료: {output_path}")
            return output_path
        else:
            print("⚠️ 트리밍 실패, 원본 파일 사용")
            # 디버그 출력 일부
            print(res.stdout[-500:])
            return input_path
    except Exception as e:
        print(f"⚠️ 트리밍 중 오류: {e}")
        return input_path

# ====== 함수 정의 ======
def test_health_check():
    """헬스체크 테스트"""
    # ENDPOINT_ID가 전체 URL인지 확인
    if ENDPOINT_ID.startswith("http"):
        url = ENDPOINT_ID.replace("/run", "/runsync")
    else:
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
            result = unwrap_runpod_output(response.json())
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
    # ENDPOINT_ID가 전체 URL인지 확인
    if ENDPOINT_ID.startswith("http"):
        url = ENDPOINT_ID.replace("/run", "/runsync")
    else:
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
            result = unwrap_runpod_output(response.json())
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

    # 0. 옵션에 따라 파일 트리밍
    voice_path = VOICE_AUDIO_PATH
    inst_path = INSTRUMENT_AUDIO_PATH
    if TRIM_SECONDS and TRIM_SECONDS > 0:
        print(f"✂️ 전송 전 트리밍 적용: 앞부분 {TRIM_SECONDS}초")
        voice_path = maybe_trim_audio(voice_path, TRIM_SECONDS)
        inst_path = maybe_trim_audio(inst_path, TRIM_SECONDS)

    # 1. 입력 오디오 → base64 인코딩
    print("🎵 오디오 파일 인코딩 중...")
    try:
        voice_audio = encode_audio_to_base64(voice_path)
        instrument_audio = encode_audio_to_base64(inst_path)
        print(f"✅ 인코딩 완료")
        print(f"   - 보컬 파일 크기: {len(voice_audio)} chars")
        print(f"   - 악기 파일 크기: {len(instrument_audio)} chars")
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        return

    # 2. API 요청 구성
    # ENDPOINT_ID가 전체 URL인지 확인
    if ENDPOINT_ID.startswith("http"):
        url = ENDPOINT_ID.replace("/run", "/runsync")
    else:
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
        result_raw = response.json()
        result = unwrap_runpod_output(result_raw)
        print("✅ 응답 수신 완료")

        if isinstance(result, dict) and "error" in result:
            print(f"❌ 서버 에러: {result['error']}")
            if "traceback" in result:
                print(f"상세 에러: {result['traceback']}")
            return

        # 출력 오디오 추출
        if isinstance(result, dict) and "output_audio" in result:
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
    # 커맨드 라인 인자 파싱
    args = parse_arguments()
    
    # 전역 변수 설정
    API_KEY = args.api_key
    ENDPOINT_ID = args.endpoint_id
    VOICE_AUDIO_PATH = args.voice_audio
    INSTRUMENT_AUDIO_PATH = args.instrument_audio
    VOICE_MODEL = args.voice_model
    PITCH_ADJUST = args.pitch_adjust
    OUTPUT_FORMAT = args.output_format
    TRIM_SECONDS = args.trim_seconds
    
    print("🎵 RunPod Serverless AICoverGen API 테스트 시작")
    print("="*60)
    print(f"📋 설정 정보:")
    print(f"   - Endpoint ID: {ENDPOINT_ID}")
    print(f"   - 보이스 모델: {VOICE_MODEL}")
    print(f"   - 피치 조정: {PITCH_ADJUST}")
    print(f"   - 출력 포맷: {OUTPUT_FORMAT}")
    print(f"   - 보컬 파일: {VOICE_AUDIO_PATH}")
    print(f"   - 악기 파일: {INSTRUMENT_AUDIO_PATH}")
    if TRIM_SECONDS and TRIM_SECONDS > 0:
        print(f"   - 트리밍: 앞부분 {TRIM_SECONDS}초 전송")
    print("="*60)

    # 1. 헬스체크
    if not test_health_check():
        print("❌ 서버가 정상 작동하지 않습니다. 종료합니다.")
        sys.exit(1)

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

    # 3. 테스트 모드에 따른 실행
    if args.test_mode == "basic":
        test_generate_cover()
    elif args.test_mode == "multiple-models":
        test_multiple_models()
    elif args.test_mode == "pitch-variations":
        test_pitch_variations()
    elif args.test_mode == "different-formats":
        test_different_formats()

    print("\n🎉 모든 테스트 완료!")
