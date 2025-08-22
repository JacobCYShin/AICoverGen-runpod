""" AICoverGen Handler for RunPod Serverless """

import os
import json
import base64
import tempfile
import logging
import shutil
from typing import Dict, Any, Optional
import traceback

import runpod
import torch
import numpy as np
import gc
import hashlib
import requests
import time
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# AICoverGen imports
import sys

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

try:
    from rvc import Config, load_hubert, get_vc, rvc_infer
    from webui import get_current_models, rvc_models_dir, mdxnet_models_dir, output_dir
except ImportError as e:
    print(f"AICoverGen 모듈 import 실패: {e}")
    raise

# Import main module to override its paths at runtime
import main as main_module

# Override paths for RunPod Serverless
RUNPOD_RVC_MODELS_DIR = "/runpod-volume/rvc_models"
RUNPOD_MDXNET_MODELS_DIR = "/runpod-volume/mdxnet_models"
RUNPOD_OUTPUT_DIR = "/runpod-volume/output"

# Audio processing imports
try:
    from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
    from pedalboard.io import AudioFile
    from pydub import AudioSegment
    import soundfile as sf
    import sox
except ImportError as e:
    print(f"Audio processing 모듈 import 실패: {e}")
    raise

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS S3 설정 (환경변수에서 가져오기)
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'likebutter-bucket')

# S3 사용 가능 여부 체크
S3_AVAILABLE = bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME)

# S3 클라이언트 초기화
def get_s3_client():
    """S3 클라이언트를 반환합니다."""
    try:
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            logger.warning("AWS 인증 정보가 설정되지 않았습니다. 환경변수를 확인해주세요.")
            return None
            
        return boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
    except Exception as e:
        logger.error(f"S3 클라이언트 생성 실패: {e}")
        return None

# RunPod 업로드 유틸리티 (URL 반환용)
try:
    from runpod.serverless.utils import rp_upload
except Exception:  # 로컬 환경 대비
    rp_upload = None

# Ensure directories exist
os.makedirs(RUNPOD_RVC_MODELS_DIR, exist_ok=True)
os.makedirs(RUNPOD_MDXNET_MODELS_DIR, exist_ok=True)
os.makedirs(RUNPOD_OUTPUT_DIR, exist_ok=True)

# Check if base models exist in volume
def check_base_models_in_volume():
    """볼륨에 기본 모델 파일들이 존재하는지 확인"""
    try:
        required_models = ["hubert_base.pt", "rmvpe.pt"]
        missing_models = []
        
        for fname in required_models:
            model_path = os.path.join(RUNPOD_RVC_MODELS_DIR, fname)
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                logger.info(f"기본 모델 확인: {fname} ({file_size:.1f}MB)")
                
                # 파일 크기 검증
                if fname == "hubert_base.pt" and file_size < 80:
                    logger.warning(f"hubert_base.pt 파일이 너무 작음 ({file_size:.1f}MB)")
                    missing_models.append(fname)
                elif fname == "rmvpe.pt" and file_size < 10:
                    logger.warning(f"rmvpe.pt 파일이 너무 작음 ({file_size:.1f}MB)")
                    missing_models.append(fname)
            else:
                logger.warning(f"기본 모델 파일이 없음: {model_path}")
                missing_models.append(fname)
        
        if missing_models:
            logger.warning(f"누락된 기본 모델: {missing_models}")
            return False
        else:
            logger.info("모든 기본 모델이 정상적으로 존재함")
            return True
            
    except Exception as e:
        logger.warning(f"기본 모델 확인 중 오류: {e}")
        return False

check_base_models_in_volume()

# S3 설정 상태 로깅 (상세 디버깅)
logger.info("🔍 환경변수 디버깅:")
logger.info(f"  AWS_ACCESS_KEY_ID: {'설정됨' if AWS_ACCESS_KEY_ID else '없음'} (길이: {len(AWS_ACCESS_KEY_ID) if AWS_ACCESS_KEY_ID else 0})")
logger.info(f"  AWS_SECRET_ACCESS_KEY: {'설정됨' if AWS_SECRET_ACCESS_KEY else '없음'} (길이: {len(AWS_SECRET_ACCESS_KEY) if AWS_SECRET_ACCESS_KEY else 0})")
logger.info(f"  AWS_REGION: {AWS_REGION}")
logger.info(f"  S3_BUCKET_NAME: {S3_BUCKET_NAME}")

if S3_AVAILABLE:
    logger.info(f"✅ S3 연결 가능: 버킷={S3_BUCKET_NAME}, 리전={AWS_REGION}")
else:
    logger.warning("⚠️ S3 연결 불가: AWS 자격증명 또는 버킷명이 설정되지 않음. base64 방식만 사용 가능.")
    
    # 모든 환경변수 출력 (디버깅용)
    logger.warning("전체 환경변수 목록:")
    for key, value in os.environ.items():
        if 'AWS' in key or 'S3' in key:
            logger.warning(f"  {key}: {'설정됨' if value else '없음'}")

def _download_from_s3(s3_url: str) -> str:
    """S3 URL에서 파일을 다운로드하여 임시 파일 경로를 반환합니다."""
    try:
        logger.info(f"S3 URL 다운로드 시작: {s3_url}")
        
        # S3 URL 파싱 - 다양한 형식 지원
        # 형식 1: https://bucket.s3.region.amazonaws.com/key
        # 형식 2: https://s3.region.amazonaws.com/bucket/key  
        # 형식 3: https://bucket.s3.amazonaws.com/key (us-east-1)
        
        if not s3_url.startswith('https://'):
            raise ValueError(f"Invalid S3 URL format: {s3_url}")
            
        url_without_protocol = s3_url.replace('https://', '')
        url_parts = url_without_protocol.split('/')
        
        if len(url_parts) < 2:
            raise ValueError(f"Invalid S3 URL format: {s3_url}")
        
        # 도메인 부분과 경로 부분 분리
        domain_part = url_parts[0]
        path_parts = url_parts[1:]
        
        if 's3' in domain_part:
            if domain_part.startswith('s3.'):
                # 형식 2: s3.region.amazonaws.com/bucket/key
                bucket_name = path_parts[0]
                s3_key = '/'.join(path_parts[1:])
            else:
                # 형식 1, 3: bucket.s3.region.amazonaws.com/key
                bucket_name = domain_part.split('.')[0]
                s3_key = '/'.join(path_parts)
        else:
            raise ValueError(f"Not a valid S3 URL: {s3_url}")
            
        if not bucket_name or not s3_key:
            raise ValueError(f"Could not parse bucket or key from URL: {s3_url}")
        
        logger.info(f"파싱된 S3 정보: bucket={bucket_name}, key={s3_key}")
        
        # S3 클라이언트 가져오기
        s3_client = get_s3_client()
        if s3_client is None:
            raise RuntimeError("S3 클라이언트를 초기화할 수 없습니다. AWS 자격증명을 확인해주세요.")
        
        # 임시 파일 생성 - 확장자 추출 시도
        file_extension = '.wav'  # 기본값
        if '.' in s3_key:
            file_extension = '.' + s3_key.split('.')[-1]
            
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_file_path = temp_file.name
        temp_file.close()
        
        # S3에서 다운로드
        logger.info(f"S3 다운로드 시작: {bucket_name}/{s3_key} -> {temp_file_path}")
        s3_client.download_file(bucket_name, s3_key, temp_file_path)
        
        # 다운로드된 파일 크기 확인
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"S3 다운로드 완료: {temp_file_path} ({file_size} bytes)")
        
        return temp_file_path
        
    except Exception as e:
        logger.error(f"S3 다운로드 실패: {s3_url} - {e}")
        # 상세한 에러 정보 로깅
        logger.error(f"에러 타입: {type(e).__name__}")
        if hasattr(e, 'response'):
            logger.error(f"AWS 응답: {e.response}")
        raise

def _upload_to_s3(file_path: str, file_type: str = "audio") -> str:
    """파일을 S3에 업로드하고 공개 URL을 반환합니다."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"업로드할 파일이 존재하지 않습니다: {file_path}")
            
        s3_client = get_s3_client()
        if s3_client is None:
            raise RuntimeError("S3 클라이언트를 초기화할 수 없습니다. AWS 자격증명을 확인해주세요.")
        
        # 파일 크기 확인
        file_size = os.path.getsize(file_path)
        logger.info(f"S3 업로드 시작: {file_path} ({file_size} bytes)")
        
        # S3 키 생성 (타임스탬프 포함)
        timestamp = int(time.time())
        file_name = os.path.basename(file_path)
        
        if file_type == "audio":
            s3_key = f"generated-audios/{timestamp}_{file_name}"
        else:
            s3_key = f"generated-images/{timestamp}_{file_name}"
        
        # S3에 업로드
        logger.info(f"S3 업로드 중: {S3_BUCKET_NAME}/{s3_key}")
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        
        # S3 공개 URL 생성
        s3_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        
        logger.info(f"S3 업로드 완료: {file_path} -> {s3_url}")
        return s3_url
        
    except Exception as e:
        logger.error(f"S3 업로드 실패: {file_path} - {e}")
        # 상세한 에러 정보 로깅
        logger.error(f"에러 타입: {type(e).__name__}")
        if hasattr(e, 'response'):
            logger.error(f"AWS 응답: {e.response}")
        raise

def _encode_outputs_as_base64(file_paths: list[str]) -> Dict[str, str]:
    """출력 파일을 base64로 인코딩하여 반환합니다."""
    result_files: Dict[str, str] = {}
    for output_file in file_paths:
        if os.path.exists(output_file):
            try:
                with open(output_file, "rb") as f:
                    file_data = f.read()
                    file_name = os.path.basename(output_file)
                    
                    # 파일 크기 확인 및 압축 고려
                    file_size = len(file_data)
                    logger.info(f"파일 크기: {file_name} = {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
                    
                    # 파일이 너무 크면 경고
                    if file_size > 50 * 1024 * 1024:  # 50MB
                        logger.warning(f"파일이 너무 큽니다: {file_name} ({file_size / 1024 / 1024:.2f} MB)")
                        # 파일을 건너뛰고 경고만 반환
                        result_files[f"{file_name}_SKIPPED"] = "File too large"
                        continue
                    
                    result_files[file_name] = base64.b64encode(file_data).decode('utf-8')
                    logger.info(f"파일 인코딩 완료: {file_name}")
            except Exception as e:
                logger.error(f"파일 인코딩 실패: {output_file} - {e}")
                result_files[f"{os.path.basename(output_file)}_ERROR"] = str(e)
        else:
            logger.warning(f"파일이 존재하지 않습니다: {output_file}")
    return result_files

def _upload_outputs_and_get_urls(file_paths: list[str]) -> Dict[str, str]:
    """출력 파일을 업로드하고 공개 URL을 반환합니다."""
    if rp_upload is None:
        raise RuntimeError("rp_upload 모듈을 사용할 수 없습니다. 런포드 서버리스 환경에서 실행해 주세요.")

    uploaded_files: Dict[str, str] = {}
    for output_file in file_paths:
        if os.path.exists(output_file):
            try:
                upload_result = rp_upload.upload_file(output_file)
                # upload_result 예: { 'file_id': str, 'url'|'link': str }
                url_value = upload_result.get('url') or upload_result.get('link')
                uploaded_files[os.path.basename(output_file)] = url_value
                logger.info(f"업로드 완료: {output_file} -> {url_value}")
            except Exception as e:
                logger.error(f"파일 업로드 실패: {output_file} - {e}")
                raise
        else:
            logger.warning(f"파일이 존재하지 않습니다: {output_file}")
    return uploaded_files

# 전역 변수로 AICoverGenHandler 인스턴스 저장 (Cold start 최적화)
aicovergen_handler = None

def discover_voice_models(models_root: str) -> list:
    """Discover valid voice model directories that contain at least one .pth file.
    Hidden items and non-directories are ignored.
    """
    if not os.path.isdir(models_root):
        return []
    discovered = []
    try:
        for name in os.listdir(models_root):
            if name.startswith('.'):
                continue
            full_path = os.path.join(models_root, name)
            if not os.path.isdir(full_path):
                continue
            try:
                if any(fname.endswith('.pth') for fname in os.listdir(full_path)):
                    discovered.append(name)
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"모델 디렉터리 스캔 중 경고: {e}")
    return discovered


def load_aicovergen_handler():
    """AICoverGenHandler 인스턴스를 로드하고 모델을 준비합니다."""
    global aicovergen_handler
    if aicovergen_handler is None:
        try:
            logger.info("AICoverGenHandler 인스턴스를 초기화하고 모델을 로드합니다...")
            aicovergen_handler = AICoverGenHandler()
            logger.info("AICoverGenHandler 초기화 완료")
        except Exception as e:
            logger.error(f"AICoverGenHandler 초기화 실패: {str(e)}")
            raise
    return aicovergen_handler

class AICoverGenHandler:
    def __init__(self):
        """Initialize the AICoverGen handler"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load available models from RunPod volume (filtered)
        self.voice_models = discover_voice_models(RUNPOD_RVC_MODELS_DIR)
        logger.info(f"Available voice models: {self.voice_models}")
        
        # Set torch to use memory efficiently for serverless
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "device": self.device,
            "available_models": self.voice_models,
            "gpu_available": torch.cuda.is_available(),
            "s3_available": S3_AVAILABLE,
            "s3_bucket": S3_BUCKET_NAME if S3_AVAILABLE else None,
            "aws_env_debug": {
                "AWS_ACCESS_KEY_ID": "설정됨" if AWS_ACCESS_KEY_ID else "없음",
                "AWS_SECRET_ACCESS_KEY": "설정됨" if AWS_SECRET_ACCESS_KEY else "없음", 
                "AWS_REGION": AWS_REGION,
                "S3_BUCKET_NAME": S3_BUCKET_NAME
            },
            "supported_input_types": ["voice_audio_url", "instrument_audio_url"] if S3_AVAILABLE else ["voice_audio", "instrument_audio"],
            "supported_return_types": ["url", "base64"] if S3_AVAILABLE else ["base64"],
            "code_version": "backup_vocals_removed_v2.1",  # 코드 변경 확인용 마커
            "backup_vocals_removed": True,  # 백업 보컬 제거 확인
            "combine_audio_signature": "main_gain_inst_gain_only"  # 함수 시그니처 확인
        }
    
    def list_models(self) -> Dict[str, Any]:
        """List available voice models"""
        return {
            "models": self.voice_models,
            "count": len(self.voice_models)
        }
    
    def check_model_files(self) -> Dict[str, Any]:
        """Check the status of model files"""
        try:
            model_status = {}
            
            # Check RVC base models
            rvc_base_models = ["hubert_base.pt", "rmvpe.pt"]
            for model in rvc_base_models:
                model_path = os.path.join(RUNPOD_RVC_MODELS_DIR, model)
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                    model_status[model] = {
                        "exists": True,
                        "size_mb": round(file_size, 2),
                        "path": model_path
                    }
                else:
                    model_status[model] = {
                        "exists": False,
                        "size_mb": 0,
                        "path": model_path
                    }
            
            # Check voice models
            voice_model_status = {}
            for voice_model in self.voice_models:
                model_dir = os.path.join(RUNPOD_RVC_MODELS_DIR, voice_model)
                if os.path.exists(model_dir):
                    files = os.listdir(model_dir)
                    voice_model_status[voice_model] = {
                        "exists": True,
                        "files": files,
                        "path": model_dir
                    }
                else:
                    voice_model_status[voice_model] = {
                        "exists": False,
                        "files": [],
                        "path": model_dir
                    }
            
            return {
                "rvc_base_models": model_status,
                "voice_models": voice_model_status,
                "runpod_volume_exists": os.path.exists("/runpod-volume"),
                "runpod_rvc_models_dir": RUNPOD_RVC_MODELS_DIR
            }
        except Exception as e:
            logger.error(f"모델 파일 상태 확인 중 오류: {str(e)}")
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def get_rvc_model(self, voice_model: str) -> tuple:
        """Get RVC model paths"""
        rvc_model_filename, rvc_index_filename = None, None
        model_dir = os.path.join(RUNPOD_RVC_MODELS_DIR, voice_model)
        
        for file in os.listdir(model_dir):
            ext = os.path.splitext(file)[1]
            if ext == '.pth':
                rvc_model_filename = file
            if ext == '.index':
                rvc_index_filename = file

        if rvc_model_filename is None:
            raise Exception(f'No model file exists in {model_dir}.')

        return os.path.join(model_dir, rvc_model_filename), os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ''
    
    def voice_change(self, voice_model: str, vocals_path: str, output_path: str, 
                    pitch_change: int, f0_method: str, index_rate: float, 
                    filter_radius: int, rms_mix_rate: float, protect: float, 
                    crepe_hop_length: int):
        """Convert voice using RVC - using main.py logic"""
        # Override all module paths to point to RunPod volume
        main_module.rvc_models_dir = RUNPOD_RVC_MODELS_DIR
        main_module.mdxnet_models_dir = RUNPOD_MDXNET_MODELS_DIR
        main_module.output_dir = RUNPOD_OUTPUT_DIR
        
        # Also override webui module paths
        import webui
        webui.rvc_models_dir = RUNPOD_RVC_MODELS_DIR
        webui.mdxnet_models_dir = RUNPOD_MDXNET_MODELS_DIR
        webui.output_dir = RUNPOD_OUTPUT_DIR
        
        # Set environment variables for other modules that use BASE_DIR
        os.environ['RVC_MODELS_DIR'] = RUNPOD_RVC_MODELS_DIR
        os.environ['MDXNET_MODELS_DIR'] = RUNPOD_MDXNET_MODELS_DIR
        os.environ['OUTPUT_DIR'] = RUNPOD_OUTPUT_DIR
        
        # Override BASE_DIR for modules that use it
        import sys
        if 'src' in sys.modules:
            src_module = sys.modules['src']
            if hasattr(src_module, 'BASE_DIR'):
                src_module.BASE_DIR = RUNPOD_RVC_MODELS_DIR
        
        # Call main.py's voice_change
        main_module.voice_change(voice_model, vocals_path, output_path, pitch_change, 
                                 f0_method, index_rate, filter_radius, rms_mix_rate, 
                                 protect, crepe_hop_length, is_webui=False)
    
    def add_audio_effects(self, audio_path: str, reverb_rm_size: float, 
                         reverb_wet: float, reverb_dry: float, reverb_damping: float) -> str:
        """Apply audio effects to vocals - using main.py logic"""
        output_path = f'{os.path.splitext(audio_path)[0]}_mixed.wav'

        # Initialize audio effects plugins - same as main.py
        board = Pedalboard(
            [
                HighpassFilter(),
                Compressor(ratio=4, threshold_db=-15),
                Reverb(room_size=reverb_rm_size, dry_level=reverb_dry, wet_level=reverb_wet, damping=reverb_damping)
             ]
        )

        with AudioFile(audio_path) as f:
            with AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
                # Read one second of audio at a time, until the file is empty:
                while f.tell() < f.frames:
                    chunk = f.read(int(f.samplerate))
                    effected = board(chunk, f.samplerate, reset=False)
                    o.write(effected)

        return output_path
    
    def combine_audio(self, audio_paths: list, output_path: str, main_gain: int, 
                     inst_gain: int, output_format: str):
        """Combine AI vocals and instrumentals - using main.py logic"""
        # 디버깅 정보 출력
        logger.info(f"🎵 오디오 파일 결합 중:")
        logger.info(f"   보컬 파일: {audio_paths[0]}")
        logger.info(f"   악기 파일: {audio_paths[1]}")
        logger.info(f"   출력 경로: {output_path}")
        logger.info(f"   main_gain: {main_gain}, inst_gain: {inst_gain}")
        
        try:
            # 파일 형식 자동 감지 (WAV/MP3 모두 지원)
            main_vocal_audio = AudioSegment.from_file(audio_paths[0]) - 4 + main_gain
            instrumental_audio = AudioSegment.from_file(audio_paths[1]) - 7 + inst_gain
            main_vocal_audio.overlay(instrumental_audio).export(output_path, format=output_format)
            logger.info(f"✅ 오디오 결합 완료: {output_path}")
        except Exception as e:
            logger.error(f"❌ 오디오 결합 실패: {str(e)}")
            # 파일 존재 여부 확인
            for i, path in enumerate(audio_paths):
                exists = os.path.exists(path) if path.startswith('/') else True  # S3 URL인 경우 존재한다고 가정
                logger.error(f"   파일 {i}: {path} - 존재: {exists}")
            raise
    
    def pitch_shift(self, audio_path: str, pitch_change: int) -> str:
        """Apply pitch shift to audio - using main.py logic"""
        return main_module.pitch_shift(audio_path, pitch_change)
    
    def generate_cover_from_separate_audio(self, 
                                         voice_audio: Optional[str] = None,  # base64 encoded voice audio (deprecated)
                                         instrument_audio: Optional[str] = None,  # base64 encoded instrument audio (deprecated)
                                         voice_audio_url: Optional[str] = None,  # S3 URL for voice audio (preferred)
                                         instrument_audio_url: Optional[str] = None,  # S3 URL for instrument audio (preferred)
                                         voice_model: str = '',
                                         pitch_adjust: int = 0,
                                         index_rate: float = 0.75,
                                         filter_radius: int = 3,
                                         rms_mix_rate: float = 0.25,
                                         protect: float = 0.33,
                                         f0_method: str = "rmvpe",
                                         crepe_hop_length: int = 128,
                                         pitch_change_all: int = 0,
                                         reverb_rm_size: float = 0.15,
                                         reverb_wet: float = 0.2,
                                         reverb_dry: float = 0.8,
                                         reverb_damping: float = 0.7,
                                         main_gain: int = 0,

                                         inst_gain: int = 0,
                                         output_format: str = "mp3",
                                         return_type: str = "url",  # 'url' | 'base64', default to 'url'
                                         **kwargs) -> Dict[str, Any]:
        """
        Generate AI cover from separate voice and instrument audio files
        
        Args:
            voice_audio: Base64 encoded voice audio file (deprecated, use voice_audio_url)
            instrument_audio: Base64 encoded instrument audio file (deprecated, use instrument_audio_url)
            voice_audio_url: S3 URL for voice audio (preferred method)
            instrument_audio_url: S3 URL for instrument audio (preferred method)
            voice_model: Name of the voice model to use
            pitch_adjust: Pitch adjustment for voice conversion
            index_rate: Index rate for voice conversion
            filter_radius: Filter radius
            rms_mix_rate: RMS mix rate
            protect: Protection value
            f0_method: F0 method
            crepe_hop_length: Crepe hop length
            pitch_change_all: Overall pitch change for all audio
            reverb_rm_size: Reverb room size
            reverb_wet: Reverb wet level
            reverb_dry: Reverb dry level
            reverb_damping: Reverb damping
            main_gain: Volume change for AI main vocals

            inst_gain: Volume change for instrumentals
            output_format: Output format of audio file
            return_type: 'url' for S3 URL output, 'base64' for base64 output
            
        Returns:
            Dict containing the generated audio file (URL or base64) and metadata
        """
        try:
            # Validate voice model
            if voice_model not in self.voice_models:
                return {
                    "error": f"Voice model '{voice_model}' not found. Available models: {self.voice_models}"
                }
            
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Prepare file paths (will be downloaded from S3 or decoded from base64)
                voice_path = None
                instrument_path = None

                # Temporary files to clean up later
                temp_files_to_cleanup = []

                # Handle voice audio input - prefer S3 URL
                if voice_audio_url:
                    if not S3_AVAILABLE:
                        return {"error": "S3 URL이 제공되었지만 S3 자격증명이 설정되지 않았습니다. AWS 환경변수를 확인해주세요."}
                    logger.info("[~] Downloading voice audio from S3...")
                    voice_path = _download_from_s3(voice_audio_url)
                    temp_files_to_cleanup.append(voice_path)
                elif voice_audio:
                    logger.info("[~] Decoding voice audio from base64...")
                    try:
                        voice_data = base64.b64decode(voice_audio)
                        voice_path = os.path.join(temp_dir, "voice_input.wav")
                        with open(voice_path, 'wb') as f:
                            f.write(voice_data)
                    except Exception as e:
                        return {"error": f"Invalid base64 voice audio data: {str(e)}"}
                else:
                    return {"error": "Missing voice audio. Provide 'voice_audio_url' (S3 URL) or 'voice_audio' (base64)."}

                # Handle instrument audio input - prefer S3 URL
                if instrument_audio_url:
                    if not S3_AVAILABLE:
                        return {"error": "S3 URL이 제공되었지만 S3 자격증명이 설정되지 않았습니다. AWS 환경변수를 확인해주세요."}
                    logger.info("[~] Downloading instrument audio from S3...")
                    instrument_path = _download_from_s3(instrument_audio_url)
                    temp_files_to_cleanup.append(instrument_path)
                elif instrument_audio:
                    logger.info("[~] Decoding instrument audio from base64...")
                    try:
                        instrument_data = base64.b64decode(instrument_audio)
                        instrument_path = os.path.join(temp_dir, "instrument_input.wav")
                        with open(instrument_path, 'wb') as f:
                            f.write(instrument_data)
                    except Exception as e:
                        return {"error": f"Invalid base64 instrument audio data: {str(e)}"}
                else:
                    return {"error": "Missing instrument audio. Provide 'instrument_audio_url' (S3 URL) or 'instrument_audio' (base64)."}
                
                # Generate unique song ID from file contents
                hasher = hashlib.blake2b(digest_size=6)
                for p in [voice_path, instrument_path]:
                    with open(p, 'rb') as fh:
                        for chunk in iter(lambda: fh.read(8192), b''):
                            hasher.update(chunk)
                song_id = hasher.hexdigest()
                song_dir = os.path.join(temp_dir, song_id)
                os.makedirs(song_dir, exist_ok=True)
                
                # Voice conversion (main.py 289-292 lines equivalent)
                pitch_change = pitch_adjust * 12 + pitch_change_all
                ai_vocals_path = os.path.join(song_dir, f'voice_{voice_model}_p{pitch_change}_i{index_rate}_fr{filter_radius}_rms{rms_mix_rate}_pro{protect}_{f0_method}{"" if f0_method != "mangio-crepe" else f"_{crepe_hop_length}"}.wav')
                
                logger.info('[~] Converting voice using RVC...')
                self.voice_change(voice_model, voice_path, ai_vocals_path, pitch_change, 
                                  f0_method, index_rate, filter_radius, rms_mix_rate, 
                                  protect, crepe_hop_length)
                
                # Apply audio effects to vocals (main.py 294-295 lines equivalent)
                logger.info('[~] Applying audio effects to Vocals...')
                ai_vocals_mixed_path = self.add_audio_effects(ai_vocals_path, reverb_rm_size, 
                                                              reverb_wet, reverb_dry, reverb_damping)
                
                # Apply overall pitch change if needed (main.py 297-300 lines equivalent)
                if pitch_change_all != 0:
                    logger.info('[~] Applying overall pitch change')
                    instrument_path = self.pitch_shift(instrument_path, pitch_change_all)
                
                # Combine AI vocals and instrumentals (main.py 302-303 lines equivalent)
                logger.info('[~] Combining AI Vocals and Instrumentals...')
                ai_cover_path_wav = os.path.join(song_dir, f'cover_{voice_model}.wav')
                self.combine_audio([ai_vocals_mixed_path, instrument_path], 
                                   ai_cover_path_wav, main_gain, inst_gain, 'wav')
                
                # Convert to MP3 if requested
                if (output_format or '').lower() == 'mp3':
                    logger.info('[~] Converting WAV to MP3...')
                    final_output_path = os.path.join(song_dir, f'cover_{voice_model}.mp3')
                    audio_seg = AudioSegment.from_file(ai_cover_path_wav)
                    audio_seg.export(final_output_path, format='mp3')
                else:
                    final_output_path = ai_cover_path_wav
                
                # Prepare result based on return_type
                result = {
                    "success": True,
                    "filename": os.path.basename(final_output_path),
                    "size": os.path.getsize(final_output_path),
                    "model_used": voice_model,
                    "return_type": return_type,
                    "parameters": {
                        "pitch_adjust": pitch_adjust,
                        "index_rate": index_rate,
                        "filter_radius": filter_radius,
                        "rms_mix_rate": rms_mix_rate,
                        "protect": protect,
                        "f0_method": f0_method,
                        "pitch_change_all": pitch_change_all,
                        "reverb_rm_size": reverb_rm_size,
                        "reverb_wet": reverb_wet,
                        "reverb_dry": reverb_dry,
                        "reverb_damping": reverb_damping,
                        "main_gain": main_gain,

                        "inst_gain": inst_gain,
                        "output_format": (output_format or 'wav')
                    }
                }

                # Handle output based on return_type
                try:
                    if return_type == "url":
                        # Upload to S3 and return URL
                        if not S3_AVAILABLE:
                            logger.warning("S3가 사용 불가능하므로 base64로 대체합니다.")
                            # Fallback to base64
                            with open(final_output_path, 'rb') as f:
                                audio_bytes = f.read()
                                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                            result["output_audio"] = audio_b64
                            result["return_type"] = "base64"
                            logger.info("[+] Base64 encoding completed (S3 fallback)")
                        else:
                            logger.info("[~] Uploading result to S3...")
                            output_url = _upload_to_s3(final_output_path, "audio")
                            result["output_url"] = output_url
                            logger.info(f"[+] S3 upload completed: {output_url}")
                    else:
                        # Return as base64 (fallback)
                        logger.info("[~] Encoding result as base64...")
                        with open(final_output_path, 'rb') as f:
                            audio_bytes = f.read()
                            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                        result["output_audio"] = audio_b64
                        logger.info("[+] Base64 encoding completed")
                        
                finally:
                    # Clean up temporary S3 downloaded files
                    for temp_file in temp_files_to_cleanup:
                        try:
                            if os.path.exists(temp_file):
                                os.unlink(temp_file)
                                logger.info(f"Cleaned up temp file: {temp_file}")
                        except Exception as e:
                            logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
                
                # Clean up GPU memory after successful generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                return result
                        
        except Exception as e:
            logger.error(f"Error during cover generation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Clean up temporary S3 downloaded files in case of error
            if 'temp_files_to_cleanup' in locals():
                for temp_file in temp_files_to_cleanup:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                            logger.info(f"Cleaned up temp file on error: {temp_file}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temp file on error {temp_file}: {cleanup_error}")
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return {
                "error": f"Error during cover generation: {str(e)}",
                "traceback": traceback.format_exc()
            }

def handler(job):
    """
    RunPod Serverless 핸들러 함수
    
    Args:
        job: RunPod에서 전달하는 작업 데이터
        
    Returns:
        Dict: 처리 결과
    """
    try:
        job_input = job.get("input", {})
        logger.info(f"작업 입력: {job_input}")
        
        # 작업 타입 확인
        operation = job_input.get("operation", "generate_cover_from_separate_audio")
        
        logger.info(f"Processing operation: {operation}")
        
        # AICoverGenHandler 인스턴스 로드
        handler_instance = load_aicovergen_handler()
        
        # Route to appropriate handler method
        if operation == "health_check":
            return handler_instance.health_check()
        elif operation == "list_models":
            return handler_instance.list_models()
        elif operation == "check_model_files":
            return handler_instance.check_model_files()
        elif operation == "generate_cover_from_separate_audio":
            # Extract parameters for cover generation from separate audio files
            params = job_input.get("params", {})
            # Default to 'url' for S3-based operation, but allow override
            if "return_type" not in params:
                params["return_type"] = "url"
            
            result = handler_instance.generate_cover_from_separate_audio(**params)
            return result
        else:
            return {
                "error": f"Unknown operation: {operation}. Available operations: health_check, list_models, generate_cover_from_separate_audio"
            }
            
    except Exception as e:
        logger.error(f"핸들러 오류: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": "Internal server error",
            "message": str(e)
        }

# Cold start 최적화: 컨테이너 시작 시 모델 미리 로드
try:
    logger.info("컨테이너 시작 시 모델 미리 로드 중...")
    load_aicovergen_handler()
    logger.info("Cold start 최적화 완료")
except Exception as e:
    logger.error(f"Cold start 최적화 실패: {str(e)}")

# 로컬 테스트용 함수
def test_local():
    """로컬 테스트용 함수"""
    print("=== 로컬 테스트 시작 ===")
    
    # 헬스체크 테스트
    print("1. 헬스체크 테스트")
    try:
        handler_instance = load_aicovergen_handler()
        result = handler_instance.health_check()
        print(f"결과: {result}")
    except Exception as e:
        print(f"오류: {e}")
    
    # 모델 목록 조회 테스트
    print("2. 모델 목록 조회 테스트")
    try:
        result = handler_instance.list_models()
        print(f"결과: {result}")
    except Exception as e:
        print(f"오류: {e}")
    
    print("\n=== 로컬 테스트 완료 ===")

# RunPod Serverless 시작
if __name__ == "__main__":
    # 로컬 테스트 모드 확인
    if os.getenv("LOCAL_TEST", "false").lower() == "true":
        test_local()
    else:
        runpod.serverless.start({"handler": handler})
