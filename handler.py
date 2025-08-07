""" AICoverGen Handler for RunPod Serverless """

import runpod
import os
import json
import base64
import tempfile
import shutil
from typing import Dict, Any, Optional
import torch
import numpy as np
import gc
import hashlib

# AICoverGen imports
import sys

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from rvc import Config, load_hubert, get_vc, rvc_infer
from webui import get_current_models, rvc_models_dir, mdxnet_models_dir, output_dir

# Override paths for RunPod Serverless
RUNPOD_RVC_MODELS_DIR = "/runpod-volume/rvc_models"
RUNPOD_MDXNET_MODELS_DIR = "/runpod-volume/mdxnet_models"
RUNPOD_OUTPUT_DIR = "/runpod-volume/output"

# Audio processing imports
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment
import soundfile as sf
import sox

# Ensure directories exist
os.makedirs(RUNPOD_RVC_MODELS_DIR, exist_ok=True)
os.makedirs(RUNPOD_MDXNET_MODELS_DIR, exist_ok=True)
os.makedirs(RUNPOD_OUTPUT_DIR, exist_ok=True)

class AICoverGenHandler:
    def __init__(self):
        """Initialize the AICoverGen handler"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load available models from RunPod volume
        self.voice_models = get_current_models(RUNPOD_RVC_MODELS_DIR)
        print(f"Available voice models: {self.voice_models}")
        
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
        from main import voice_change as main_voice_change
        main_voice_change(voice_model, vocals_path, output_path, pitch_change, 
                         f0_method, index_rate, filter_radius, rms_mix_rate, 
                         protect, crepe_hop_length, is_webui=False)
    
    def add_audio_effects(self, audio_path: str, reverb_rm_size: float, 
                         reverb_wet: float, reverb_dry: float, reverb_damping: float) -> str:
        """Apply simple audio effects to vocals"""
        import librosa
        import soundfile as sf
        import numpy as np
        from pedalboard import Pedalboard, Reverb, Compressor
        
        print(f"[~] Simple audio effects: {audio_path}")
        
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
        
        print(f"[+] Simple effects applied: {output_path}")
        
        return output_path
    
    def combine_audio(self, audio_paths: list, output_path: str, main_gain: int, 
                     backup_gain: int, inst_gain: int, output_format: str):
        """Combine AI vocals and instrumentals with simple mixing"""
        import librosa
        import soundfile as sf
        import numpy as np
        
        print(f"[~] Simple mixing: {audio_paths}")
        
        # Load all audio files
        audio_signals = []
        
        for path in audio_paths:
            if os.path.exists(path):
                # Load with lower sample rate for faster processing
                y, sr = librosa.load(path, sr=22050)
                audio_signals.append(y)
            else:
                print(f"Warning: {path} not found, using silence")
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
        
        # Save the final mix
        sf.write(output_path, final_mix, 22050)
        
        print(f"[+] Simple mixing completed: {output_path}")
        print(f"    - Duration: {len(final_mix) / 22050:.2f} seconds")
        print(f"    - Peak level: {np.max(np.abs(final_mix)):.3f}")
        
        return output_path
    
    def pitch_shift(self, audio_path: str, pitch_change: int) -> str:
        """Apply pitch shift to audio - using main.py logic"""
        from main import pitch_shift as main_pitch_shift
        return main_pitch_shift(audio_path, pitch_change)
    
    def generate_cover_from_separate_audio(self, 
                                         voice_audio: str,  # base64 encoded voice audio
                                         instrument_audio: str,  # base64 encoded instrument audio
                                         voice_model: str,
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
            
        Returns:
            Dict containing the generated audio file (base64) and metadata
        """
        try:
            # Validate voice model
            if voice_model not in self.voice_models:
                return {
                    "error": f"Voice model '{voice_model}' not found. Available models: {self.voice_models}"
                }
            
            # Decode base64 audio files
            try:
                voice_data = base64.b64decode(voice_audio)
                instrument_data = base64.b64decode(instrument_audio)
            except Exception as e:
                return {"error": f"Invalid base64 audio data: {str(e)}"}
            
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save audio files to temporary directory
                voice_path = os.path.join(temp_dir, "voice_input.wav")
                instrument_path = os.path.join(temp_dir, "instrument_input.wav")
                
                with open(voice_path, 'wb') as f:
                    f.write(voice_data)
                with open(instrument_path, 'wb') as f:
                    f.write(instrument_data)
                
                # Generate unique song ID
                song_id = hashlib.blake2b(voice_data + instrument_data, digest_size=6).hexdigest()
                song_dir = os.path.join(temp_dir, song_id)
                os.makedirs(song_dir, exist_ok=True)
                
                # Voice conversion (main.py 289-292 lines equivalent)
                pitch_change = pitch_adjust * 12 + pitch_change_all
                ai_vocals_path = os.path.join(song_dir, f'voice_{voice_model}_p{pitch_change}_i{index_rate}_fr{filter_radius}_rms{rms_mix_rate}_pro{protect}_{f0_method}{"" if f0_method != "mangio-crepe" else f"_{crepe_hop_length}"}.wav')
                
                print('[~] Converting voice using RVC...')
                self.voice_change(voice_model, voice_path, ai_vocals_path, pitch_change, 
                                f0_method, index_rate, filter_radius, rms_mix_rate, 
                                protect, crepe_hop_length)
                
                # Apply audio effects to vocals (main.py 294-295 lines equivalent)
                print('[~] Applying audio effects to Vocals...')
                ai_vocals_mixed_path = self.add_audio_effects(ai_vocals_path, reverb_rm_size, 
                                                            reverb_wet, reverb_dry, reverb_damping)
                
                # Apply overall pitch change if needed (main.py 297-300 lines equivalent)
                if pitch_change_all != 0:
                    print('[~] Applying overall pitch change')
                    instrument_path = self.pitch_shift(instrument_path, pitch_change_all)
                    # For backup vocals, we'll use the original voice with pitch shift
                    backup_vocals_path = self.pitch_shift(voice_path, pitch_change_all)
                else:
                    backup_vocals_path = voice_path
                
                # Combine AI vocals and instrumentals (main.py 302-303 lines equivalent)
                print('[~] Combining AI Vocals and Instrumentals...')
                ai_cover_path = os.path.join(song_dir, f'cover_{voice_model}.{output_format}')
                self.combine_audio([ai_vocals_mixed_path, backup_vocals_path, instrument_path], 
                                 ai_cover_path, main_gain, backup_gain, inst_gain, output_format)
                
                # Read the final output file
                with open(ai_cover_path, 'rb') as f:
                    audio_bytes = f.read()
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Clean up GPU memory after successful generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                return {
                    "success": True,
                    "output_audio": audio_b64,
                    "filename": f"cover_{voice_model}.{output_format}",
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
                        "output_format": output_format
                    }
                }
                        
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return {
                "error": f"Error during cover generation: {str(e)}",
                "traceback": error_traceback
            }

# Global handler instance
handler = AICoverGenHandler()

def handler(job):
    """
    Handler function that will be used to process jobs.
    
    Args:
        job: The job object containing input data
        
    Returns:
        Dict containing the response
    """
    try:
        # Get job input
        job_input = job["input"]
        
        # Get the operation type
        operation = job_input.get("operation", "generate_cover_from_separate_audio")
        
        print(f"Processing operation: {operation}")
        
        # Route to appropriate handler method
        if operation == "health_check":
            return handler.health_check()
        elif operation == "list_models":
            return handler.list_models()
        elif operation == "generate_cover_from_separate_audio":
            # Extract parameters for cover generation from separate audio files
            params = job_input.get("params", {})
            return handler.generate_cover_from_separate_audio(**params)
        else:
            return {
                "error": f"Unknown operation: {operation}. Available operations: health_check, list_models, generate_cover_from_separate_audio"
            }
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Handler error: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return {
            "error": f"Handler error: {str(e)}",
            "traceback": error_traceback
        }

# Start the serverless handler
runpod.serverless.start({"handler": handler})
