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
            "gpu_available": torch.cuda.is_available()
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
        """Apply simple audio effects to vocals"""
        import librosa
        import soundfile as sf
        import numpy as np
        from pedalboard import Pedalboard, Reverb, Compressor
        
        logger.info(f"[~] Simple audio effects: {audio_path}")
        
        # Load audio with lower sample rate for speed
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Simple effects chain
        board = Pedalboard([
            # Light compression for vocal clarity
            Compressor(ratio=3, threshold_db=-20, attack_ms=5, release_ms=50),
            
            # Light reverb for space
            Reverb(room_size=reverb_rm_size, 
                   dry_level=reverb_dry, 
                   wet_level=reverb_wet, 
                   damping=reverb_damping)
        ])
        
        # Apply effects
        effected = board(y, sr)
        
        # Save processed audio
        output_path = f'{os.path.splitext(audio_path)[0]}_mixed.wav'
        sf.write(output_path, effected, sr)
        
        logger.info(f"[+] Simple effects applied: {output_path}")
        
        return output_path
    
    def combine_audio(self, audio_paths: list, output_path: str, main_gain: int, 
                     backup_gain: int, inst_gain: int, output_format: str):
        """Combine AI vocals and instrumentals with simple mixing"""
        import librosa
        import soundfile as sf
        import numpy as np
        
        logger.info(f"[~] Simple mixing: {audio_paths}")
        
        # Load all audio files
        audio_signals = []
        
        for path in audio_paths:
            if os.path.exists(path):
                # Load with lower sample rate for faster processing
                y, sr = librosa.load(path, sr=22050)
                audio_signals.append(y)
            else:
                logger.warning(f"Warning: {path} not found, using silence")
                audio_signals.append(np.zeros(22050 * 10))
        
        # Ensure all signals have same length
        max_length = max(len(signal) for signal in audio_signals)
        padded_signals = []
        
        for signal in audio_signals:
            if len(signal) < max_length:
                padded = np.pad(signal, (0, max_length - len(signal)), mode='constant')
            else:
                padded = signal
            padded_signals.append(padded)
        
        # Apply gain adjustments (convert dB to linear)
        main_vocals = padded_signals[0] * (10 ** (main_gain / 20))
        backup_vocals = padded_signals[1] * (10 ** (backup_gain / 20))
        instrumentals = padded_signals[2] * (10 ** (inst_gain / 20))
        
        # Simple mixing without complex processing
        # 1. Combine vocals with reduced volume
        vocals_combined = main_vocals * 0.7 + backup_vocals * 0.35  # Main at 70%, backup at 35%
        
        # 2. Simple ducking (vocals reduce instrumentals slightly)
        vocal_envelope = np.abs(vocals_combined)
        ducking_curve = 1.0 - (np.clip(vocal_envelope * 0.1, 0, 0.1))  # Duck up to 10%
        instrumentals_ducked = instrumentals * ducking_curve
        
        # 3. Final mix with proper levels
        final_mix = vocals_combined + instrumentals_ducked * 0.9  # Instrumentals at 90%
        
        # 4. Simple normalization
        max_val = np.max(np.abs(final_mix))
        if max_val > 0.95:
            final_mix = final_mix * (0.95 / max_val)
        
        # Save the final mix (always write WAV here)
        sf.write(output_path, final_mix, 22050)
        
        logger.info(f"[+] Simple mixing completed: {output_path}")
        logger.info(f"    - Duration: {len(final_mix) / 22050:.2f} seconds")
        logger.info(f"    - Peak level: {np.max(np.abs(final_mix)):.3f}")
        
        return output_path
    
    def pitch_shift(self, audio_path: str, pitch_change: int) -> str:
        """Apply pitch shift to audio - using main.py logic"""
        return main_module.pitch_shift(audio_path, pitch_change)
    
    def generate_cover_from_separate_audio(self, 
                                         voice_audio: Optional[str] = None,  # base64 encoded voice audio
                                         instrument_audio: Optional[str] = None,  # base64 encoded instrument audio
                                         voice_model: str = '',
                                         pitch_adjust: int = 0,
                                         index_rate: float = 0.5,
                                         filter_radius: int = 3,
                                         rms_mix_rate: float = 0.25,
                                         protect: float = 0.33,
                                         f0_method: str = "rmvpe",
                                         crepe_hop_length: int = 128,
                                         pitch_change_all: int = 0,
                                         reverb_rm_size: float = 0.25,
                                         reverb_wet: float = 0.4,
                                         reverb_dry: float = 0.6,
                                         reverb_damping: float = 0.5,
                                         main_gain: int = 0,
                                         backup_gain: int = 0,
                                         inst_gain: int = 0,
                                         output_format: str = "mp3",
                                         voice_audio_url: Optional[str] = None,
                                         instrument_audio_url: Optional[str] = None,
                                         **kwargs) -> Dict[str, Any]:
        """
        Generate AI cover from separate voice and instrument audio files
        
        Args:
            voice_audio: Base64 encoded voice audio file
            instrument_audio: Base64 encoded instrument audio file
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
            backup_gain: Volume change for backup vocals
            inst_gain: Volume change for instrumentals
            output_format: Output format of audio file
            voice_audio_url: URL for voice audio (alternative to base64)
            instrument_audio_url: URL for instrument audio (alternative to base64)
            
        Returns:
            Dict containing the generated audio file (base64) and metadata
        """
        try:
            # Validate voice model
            if voice_model not in self.voice_models:
                return {
                    "error": f"Voice model '{voice_model}' not found. Available models: {self.voice_models}"
                }
            
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Prepare file paths
                voice_path = os.path.join(temp_dir, "voice_input.wav")
                instrument_path = os.path.join(temp_dir, "instrument_input.wav")

                # Prefer URL inputs when provided to avoid large base64 payloads
                def download_to_file(url: str, dst: str):
                    r = requests.get(url, stream=True, timeout=120)
                    r.raise_for_status()
                    with open(dst, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                if voice_audio_url:
                    logger.info("[~] Downloading voice audio from URL...")
                    download_to_file(voice_audio_url, voice_path)
                elif voice_audio:
                    try:
                        voice_data = base64.b64decode(voice_audio)
                        with open(voice_path, 'wb') as f:
                            f.write(voice_data)
                    except Exception as e:
                        return {"error": f"Invalid base64 voice audio data: {str(e)}"}
                else:
                    return {"error": "Missing voice audio. Provide 'voice_audio' (base64) or 'voice_audio_url'."}

                if instrument_audio_url:
                    logger.info("[~] Downloading instrument audio from URL...")
                    download_to_file(instrument_audio_url, instrument_path)
                elif instrument_audio:
                    try:
                        instrument_data = base64.b64decode(instrument_audio)
                        with open(instrument_path, 'wb') as f:
                            f.write(instrument_data)
                    except Exception as e:
                        return {"error": f"Invalid base64 instrument audio data: {str(e)}"}
                else:
                    return {"error": "Missing instrument audio. Provide 'instrument_audio' (base64) or 'instrument_audio_url'."}
                
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
                    # For backup vocals, we'll use the original voice with pitch shift
                    backup_vocals_path = self.pitch_shift(voice_path, pitch_change_all)
                else:
                    backup_vocals_path = voice_path
                
                # Combine AI vocals and instrumentals (main.py 302-303 lines equivalent)
                logger.info('[~] Combining AI Vocals and Instrumentals...')
                ai_cover_path_wav = os.path.join(song_dir, f'cover_{voice_model}.wav')
                self.combine_audio([ai_vocals_mixed_path, backup_vocals_path, instrument_path], 
                                   ai_cover_path_wav, main_gain, backup_gain, inst_gain, output_format)
                
                # Convert to MP3 if requested
                if (output_format or '').lower() == 'mp3':
                    logger.info('[~] Converting WAV to MP3...')
                    final_output_path = os.path.join(song_dir, f'cover_{voice_model}.mp3')
                    audio_seg = AudioSegment.from_file(ai_cover_path_wav)
                    audio_seg.export(final_output_path, format='mp3')
                else:
                    final_output_path = ai_cover_path_wav
                
                # Read the final output file
                with open(final_output_path, 'rb') as f:
                    audio_bytes = f.read()
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Clean up GPU memory after successful generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                return {
                    "success": True,
                    "output_audio": audio_b64,
                    "filename": os.path.basename(final_output_path),
                    "size": len(audio_bytes),
                    "model_used": voice_model,
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
                        "backup_gain": backup_gain,
                        "inst_gain": inst_gain,
                        "output_format": (output_format or 'wav')
                    }
                }
                        
        except Exception as e:
            logger.error(f"Error during cover generation: {str(e)}")
            logger.error(traceback.format_exc())
            
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
            return_type = params.get("return_type", "base64")  # 'url' | 'base64'
            
            result = handler_instance.generate_cover_from_separate_audio(**params)
            
            # return_type에 따라 결과 처리
            if return_type == "url" and "output_audio" in result:
                # base64를 임시 파일로 저장하고 업로드
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                    audio_data = base64.b64decode(result["output_audio"])
                    tmp_file.write(audio_data)
                    tmp_file_path = tmp_file.name
                
                try:
                    uploaded_urls = _upload_outputs_and_get_urls([tmp_file_path])
                    result["output_urls"] = uploaded_urls
                    result["return_type"] = "url"
                    # base64 데이터 제거 (URL로 대체)
                    if "output_audio" in result:
                        del result["output_audio"]
                finally:
                    # 임시 파일 정리
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
            
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
