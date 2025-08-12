#!/usr/bin/env python3
"""
AICoverGen RunPod Serverless API 테스트 클라이언트

이 스크립트는 RunPod Serverless에 배포된 AICoverGen API를 테스트합니다.
- runsync(동기)와 run/status(비동기 폴링)를 모두 지원
- 서버 응답의 output 래핑을 해제하고, output_urls 또는 output_audio를 저장
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
from typing import Dict, Any, Optional
from urllib.parse import urlparse

class AICoverGenRunPodClient:
    """AICoverGen RunPod API 클라이언트"""

    def __init__(self, api_url: str, api_key: str = None):
        """
        클라이언트 초기화

        Args:
            api_url: RunPod Endpoint 기준 URL. 예시:
              - https://api.runpod.ai/v2/<ENDPOINT_ID>
              - 또는 기존 형식: https://api.runpod.ai/v2/<ENDPOINT_ID>/run, /runsync 중 하나
            api_key: RunPod API 키 (선택사항)
        """
        base = api_url.rstrip("/")
        if base.endswith("/run") or base.endswith("/runsync") or base.endswith("/status"):
            # 기존 형식에서 엔드포인트 베이스로 환원
            base = base.rsplit("/", 1)[0]
        self.base_url = base  # https://api.runpod.ai/v2/<ENDPOINT_ID>
        self.url_run = f"{self.base_url}/run"
        self.url_runsync = f"{self.base_url}/runsync"
        self.url_status_base = f"{self.base_url}/status"

        print(f"엔드포인트 BASE URL: {self.base_url}")
        print(f"RUN URL: {self.url_run}")
        print(f"RUNSYNC URL: {self.url_runsync}")

        self.api_key = api_key
        self.session = requests.Session()

        # 연결/재시도/타임아웃 설정
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # 5분 타임아웃 기본값
        self.session.timeout = 300

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.session.headers.update(headers)

    def _unwrap_output(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        """RunPod runsync/run 응답에서 output 래핑을 해제합니다."""
        print(f"원본 응답: {response_json}")
        
        # status가 COMPLETED인 경우 output 필드 확인
        if response_json.get("status") == "COMPLETED":
            if "output" in response_json and isinstance(response_json["output"], dict):
                print(f"output 필드 발견: {response_json['output']}")
                return response_json["output"]
            else:
                print("output 필드가 없거나 올바르지 않습니다.")
                return response_json
        
        # 일반적인 경우
        if isinstance(response_json, dict) and "output" in response_json and isinstance(response_json["output"], dict):
            return response_json["output"]
        return response_json

    def _status_url(self, job_id: str) -> str:
        return f"{self.url_status_base}/{job_id}"

    def generate_cover(
        self,
        voice_audio_path: str,
        instrument_audio_path: str,
        voice_model: str = "Jungkook",
        pitch_adjust: int = 0,
        output_format: str = "mp3",
        return_type: str = "base64",
        use_runsync: bool = True,
        poll_interval_sec: int = 5,
        max_wait_sec: int = 1800,
        **kwargs
    ) -> Dict[str, Any]:
        """
        AI 커버 생성을 수행합니다.

        Args:
            voice_audio_path: 보컬 오디오 파일 경로
            instrument_audio_path: 악기 오디오 파일 경로
            voice_model: 사용할 보이스 모델명
            pitch_adjust: 피치 조정 (-12 ~ +12)
            output_format: 출력 형식 (mp3/wav)
            return_type: "base64"(기본) 또는 "url"
            use_runsync: True면 runsync 동기 처리, False면 run+status 폴링
            poll_interval_sec: 비동기 폴링 간격
            max_wait_sec: 비동기 최대 대기시간
            **kwargs: 추가 파라미터들
        """
        # 파일 크기 안내
        voice_size = os.path.getsize(voice_audio_path)
        instrument_size = os.path.getsize(instrument_audio_path)
        print(f"보컬 파일 크기: {voice_size} bytes ({voice_size / 1024 / 1024:.2f} MB)")
        print(f"악기 파일 크기: {instrument_size} bytes ({instrument_size / 1024 / 1024:.2f} MB)")

        # 파일 읽기 및 base64 인코딩
        with open(voice_audio_path, "rb") as f:
            voice_audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        with open(instrument_audio_path, "rb") as f:
            instrument_audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        print(f"보컬(base64) 길이: {len(voice_audio_b64)} chars")
        print(f"악기(base64) 길이: {len(instrument_audio_b64)} chars")

        payload = {
            "input": {
                "operation": "generate_cover_from_separate_audio",
                "params": {
                    "voice_audio": voice_audio_b64,
                    "instrument_audio": instrument_audio_b64,
                    "voice_model": voice_model,
                    "pitch_adjust": pitch_adjust,
                    "output_format": output_format,
                    "return_type": return_type,
                    **kwargs
                }
            }
        }

        if use_runsync:
            print("runsync 요청 전송...")
            resp = self.session.post(self.url_runsync, json=payload, timeout=self.session.timeout)
            print(f"runsync 상태: {resp.status_code}")
            try:
                resp_json = resp.json()
            except Exception:
                print(f"응답 텍스트: {resp.text}")
                resp.raise_for_status()
                return {"error": "Invalid JSON"}
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}", "details": resp_json}
            # RunPod 래핑 해제
            return self._unwrap_output(resp_json)
        else:
            print("run 비동기 제출...")
            submit = self.session.post(self.url_run, json=payload, timeout=self.session.timeout)
            print(f"run 상태: {submit.status_code}")
            submit.raise_for_status()
            submit_json = submit.json()
            job_id = submit_json.get("id")
            if not job_id:
                return {"error": "No job id returned", "details": submit_json}
            print(f"작업 ID: {job_id}")

            # /status 폴링
            waited = 0
            while waited < max_wait_sec:
                status_resp = self.session.get(self._status_url(job_id), timeout=self.session.timeout)
                if status_resp.status_code != 200:
                    print(f"status HTTP {status_resp.status_code}")
                try:
                    status_json = status_resp.json()
                except Exception:
                    print(f"status 응답 텍스트: {status_resp.text}")
                    return {"error": "Invalid status JSON"}

                status = status_json.get("status") or status_json.get("state")
                print(f"상태: {status}")
                if status == "COMPLETED":
                    return self._unwrap_output(status_json)
                if status == "FAILED":
                    return {"error": "Job failed", "details": status_json}

                import time
                time.sleep(poll_interval_sec)
                waited += poll_interval_sec

            return {"error": "Timeout waiting for job completion"}

    def save_outputs(self, response_data: Dict[str, Any], output_dir: str = ".") -> bool:
        """output_audio(base64) 또는 output_urls(URL) 저장"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            saved_any = False

            # URL 저장
            if "output_urls" in response_data and isinstance(response_data["output_urls"], dict):
                for filename, url in response_data["output_urls"].items():
                    print(f"다운로드: {filename} <- {url}")
                    r = self.session.get(url, timeout=600)
                    r.raise_for_status()
                    path = os.path.join(output_dir, filename)
                    with open(path, "wb") as f:
                        f.write(r.content)
                    print(f"파일 저장됨: {path}")
                    saved_any = True

            # base64 저장
            if "output_audio" in response_data:
                filename = response_data.get("filename", f"ai_cover.{response_data.get('parameters', {}).get('output_format', 'mp3')}")
                path = os.path.join(output_dir, filename)
                with open(path, "wb") as f:
                    f.write(base64.b64decode(response_data["output_audio"]))
                print(f"파일 저장됨: {path}")
                saved_any = True

            if not saved_any:
                print("저장할 출력이 없습니다. (output_urls/output_audio 없음)")
                return False
            return True
        except Exception as e:
            print(f"파일 저장 실패: {e}")
            return False

    def test_connection(self) -> Dict[str, Any]:
        """서버 연결을 테스트합니다. run으로 'health_check' 요청."""
        try:
            payload = {"input": {"operation": "health_check"}}
            r = self.session.post(self.url_run, json=payload, timeout=30)
            try:
                j = r.json()
            except Exception:
                j = {"text": r.text}
            return {"status_code": r.status_code, "response": j}
        except Exception as e:
            return {"error": str(e)}

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
        "--voice-url",
        default=None,
        help="보컬 오디오의 공개 URL(제공 시 base64 대신 사용)"
    )
    
    parser.add_argument(
        "--instrument-url",
        default=None,
        help="악기 오디오의 공개 URL(제공 시 base64 대신 사용)"
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
VOICE_URL = None
INSTRUMENT_URL = None
VOICE_MODEL = None
PITCH_ADJUST = None
OUTPUT_FORMAT = None
TRIM_SECONDS = 0

# ====== 유틸 ======
# 사용하지 않는 함수들 제거 (클라이언트 클래스로 대체됨)

# ====== 함수 정의 ======
def test_health_check(client):
    """헬스체크 테스트"""
    print("🏥 헬스체크 테스트 중...")
    
    try:
        result = client.test_connection()
        
        if result.get("status_code") == 200:
            response_data = result.get("response", {})
            if isinstance(response_data, dict) and "output" in response_data:
                health_data = response_data["output"]
            else:
                health_data = response_data
            
            print("✅ 서버 정상 작동")
            print(f"   - 디바이스: {health_data.get('device', 'N/A')}")
            print(f"   - GPU 사용 가능: {health_data.get('gpu_available', 'N/A')}")
            print(f"   - 사용 가능한 모델: {health_data.get('available_models', [])}")
            return True
        else:
            print(f"❌ 헬스체크 실패: {result.get('status_code')}")
            print(f"응답: {result.get('response')}")
            return False
            
    except Exception as e:
        print(f"❌ 헬스체크 중 오류: {str(e)}")
        return False


def test_list_models(client):
    """사용 가능한 모델 목록 테스트"""
    print("📋 사용 가능한 모델 목록 조회 중...")
    
    try:
        # 클라이언트를 사용하여 모델 목록 조회
        payload = {"input": {"operation": "list_models"}}
        response = client.session.post(client.url_runsync, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = client._unwrap_output(response.json())
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


def test_generate_cover(client):
    """AI 커버 생성 테스트"""
    print("🚀 AI 커버 생성 요청 전송 중...")
    print(f"📊 요청 정보:")
    print(f"   - 보이스 모델: {VOICE_MODEL}")
    print(f"   - 피치 조정: {PITCH_ADJUST}")
    print(f"   - 출력 포맷: {OUTPUT_FORMAT}")
    print(f"   - 보컬 파일: {VOICE_AUDIO_PATH}")
    print(f"   - 악기 파일: {INSTRUMENT_AUDIO_PATH}")

    start_time = time.time()

    try:
        # 클라이언트를 사용하여 AI 커버 생성
        result = client.generate_cover(
            voice_audio_path=VOICE_AUDIO_PATH,
            instrument_audio_path=INSTRUMENT_AUDIO_PATH,
            voice_model=VOICE_MODEL,
            pitch_adjust=PITCH_ADJUST,
            output_format=OUTPUT_FORMAT,
            return_type="base64",
            use_runsync=True,
            index_rate=0.5,
            filter_radius=3,
            rms_mix_rate=0.25,
            protect=0.33,
            f0_method="rmvpe",
            reverb_rm_size=0.25,
            reverb_wet=0.4,
            reverb_dry=0.6,
            reverb_damping=0.5,
            main_gain=0,
            backup_gain=0,
            inst_gain=0
        )

        processing_time = time.time() - start_time
        print(f"⏱️ 처리 시간: {processing_time:.2f}초")

        if isinstance(result, dict) and "error" in result:
            print(f"❌ 서버 에러: {result['error']}")
            if "traceback" in result:
                print(f"상세 에러: {result['traceback']}")
            return

        # 출력 오디오 추출
        if isinstance(result, dict) and "output_audio" in result:
            print("✅ 응답 수신 완료")

            # 결과 저장
            output_dir = "output_results"
            if client.save_outputs(result, output_dir):
                print(f"🎉 결과 저장 완료 → {output_dir} 디렉토리")
                
                # 추가 정보 출력
                if "parameters" in result:
                    params = result["parameters"]
                    print(f"\n📋 처리 정보:")
                    for key, value in params.items():
                        print(f"   - {key}: {value}")

                print(f"📊 서버 응답 파일 크기: {result.get('size', 'N/A')} bytes")
            else:
                print("❌ 파일 저장 실패")

        else:
            print("❌ 예상된 출력 형식이 아닙니다.")
            print(f"응답 구조: {json.dumps(result, indent=2, ensure_ascii=False)}")

    except Exception as e:
        print(f"❌ 요청 중 오류: {str(e)}")
        return

# ====== 고급 테스트 함수 ======
def test_multiple_models(client):
    """여러 모델로 테스트"""
    models = ["Jungkook"]  # 사용 가능한 모델들

    for model in models:
        print(f"\n🎤 {model} 모델 테스트 중...")
        global VOICE_MODEL
        VOICE_MODEL = model
        test_generate_cover(client)
        time.sleep(3)  # 요청 간 간격


def test_pitch_variations(client):
    """다양한 피치로 테스트"""
    pitches = [-2, 0, 2]  # 낮음, 원본, 높음

    for pitch in pitches:
        print(f"\n🎵 피치 {pitch:+d} 테스트 중...")
        global PITCH_ADJUST
        PITCH_ADJUST = pitch
        test_generate_cover(client)
        time.sleep(3)


def test_different_formats(client):
    """다양한 출력 포맷 테스트"""
    formats = ["mp3", "wav"]

    for fmt in formats:
        print(f"\n🎵 {fmt.upper()} 포맷 테스트 중...")
        global OUTPUT_FORMAT
        OUTPUT_FORMAT = fmt
        test_generate_cover(client)
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
    VOICE_URL = args.voice_url
    INSTRUMENT_URL = args.instrument_url
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
    print("="*60)

    # 클라이언트 초기화
    client = AICoverGenRunPodClient(ENDPOINT_ID, API_KEY)

    # 1. 헬스체크
    if not test_health_check(client):
        print("❌ 서버가 정상 작동하지 않습니다. 종료합니다.")
        sys.exit(1)

    print("\n" + "="*60)

    # 2. 사용 가능한 모델 확인
    available_models = test_list_models(client)
    if VOICE_MODEL not in available_models:
        print(f"⚠️ 경고: '{VOICE_MODEL}' 모델이 사용 가능한 모델 목록에 없습니다.")
        if available_models:
            print(f"사용 가능한 모델 중 하나를 선택하세요: {available_models}")
            VOICE_MODEL = available_models[0]
            print(f"자동으로 '{VOICE_MODEL}' 모델을 사용합니다.")

    print("\n" + "="*60)

    # 3. 테스트 모드에 따른 실행
    if args.test_mode == "basic":
        test_generate_cover(client)
    elif args.test_mode == "multiple-models":
        test_multiple_models(client)
    elif args.test_mode == "pitch-variations":
        test_pitch_variations(client)
    elif args.test_mode == "different-formats":
        test_different_formats(client)

    print("\n🎉 모든 테스트 완료!")
