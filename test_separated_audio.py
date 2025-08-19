#!/usr/bin/env python3
"""
Test script for separated audio files in tmp/ folder
Modified from run_v2.py to work with pre-separated voice and instrumental files
"""

import argparse
import os
import sys
import tempfile
import hashlib
from pathlib import Path

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

# Import AICoverGen modules
from rvc import Config, load_hubert, get_vc, rvc_infer
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment
import gc

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
rvc_models_dir = BASE_DIR / 'rvc_models'
output_dir = BASE_DIR / 'song_output'

def get_rvc_model(voice_model):
    """Get RVC model paths"""
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = rvc_models_dir / voice_model
    
    if not model_dir.exists():
        raise Exception(f'The folder {model_dir} does not exist.')
        
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            rvc_model_filename = file
        if ext == '.index':
            rvc_index_filename = file

    if rvc_model_filename is None:
        raise Exception(f'No model file exists in {model_dir}.')

    return str(model_dir / rvc_model_filename), str(model_dir / rvc_index_filename) if rvc_index_filename else ''

def voice_change(voice_model, vocals_path, output_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length):
    """Convert voice using RVC"""
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model)
    device = 'cuda:0'
    config = Config(device, True)
    hubert_model = load_hubert(device, config.is_half, str(rvc_models_dir / 'hubert_base.pt'))
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

    # convert main vocals
    rvc_infer(rvc_index_path, index_rate, vocals_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model)
    del hubert_model, cpt
    gc.collect()

def add_audio_effects(audio_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping):
    """Apply audio effects to vocals"""
    output_path = f'{os.path.splitext(audio_path)[0]}_mixed.wav'

    # Initialize audio effects plugins
    board = Pedalboard([
        HighpassFilter(),
        Compressor(ratio=4, threshold_db=-15),
        Reverb(room_size=reverb_rm_size, dry_level=reverb_dry, wet_level=reverb_wet, damping=reverb_damping)
    ])

    with AudioFile(audio_path) as f:
        with AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)

    return output_path

def combine_audio(audio_paths, output_path, main_gain, inst_gain, output_format):
    """Combine AI vocals and instrumentals (no backup vocals)"""
    main_vocal_audio = AudioSegment.from_wav(audio_paths[0]) - 4 + main_gain
    instrumental_audio = AudioSegment.from_wav(audio_paths[1]) - 7 + inst_gain
    main_vocal_audio.overlay(instrumental_audio).export(output_path, format=output_format)

def generate_ai_cover_from_separated_files(args):
    """Generate AI cover from pre-separated voice and instrumental files"""
    
    # Validate input files
    voice_file = Path(args.voice_input)
    inst_file = Path(args.inst_input)
    
    if not voice_file.exists():
        raise FileNotFoundError(f"Voice file not found: {voice_file}")
    if not inst_file.exists():
        raise FileNotFoundError(f"Instrumental file not found: {inst_file}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Generate unique song ID from file contents
    hasher = hashlib.blake2b(digest_size=6)
    for p in [voice_file, inst_file]:
        with open(p, 'rb') as fh:
            for chunk in iter(lambda: fh.read(8192), b''):
                hasher.update(chunk)
    song_id = hasher.hexdigest()
    song_dir = output_dir / song_id
    song_dir.mkdir(exist_ok=True)
    
    print(f"[~] Processing files in: {song_dir}")
    print(f"[~] Voice model: {args.rvc_dirname}")
    
    # Calculate pitch change
    pitch_change = args.pitch_change * 12 + args.pitch_change_all
    
    # Generate AI vocals path
    ai_vocals_path = song_dir / f'ai_vocals_{args.rvc_dirname}_p{pitch_change}_i{args.index_rate}_fr{args.filter_radius}_rms{args.remix_mix_rate}_pro{args.protect}_{args.pitch_detection_algo}{"" if args.pitch_detection_algo != "mangio-crepe" else f"_{args.crepe_hop_length}"}.wav'
    
    # Voice conversion
    print('[~] Converting voice using RVC...')
    voice_change(
        args.rvc_dirname, 
        str(voice_file), 
        str(ai_vocals_path), 
        pitch_change, 
        args.pitch_detection_algo, 
        args.index_rate, 
        args.filter_radius, 
        args.remix_mix_rate, 
        args.protect, 
        args.crepe_hop_length
    )
    
    # Apply audio effects
    print('[~] Applying audio effects to Vocals...')
    ai_vocals_mixed_path = add_audio_effects(
        str(ai_vocals_path), 
        args.reverb_size, 
        args.reverb_wetness, 
        args.reverb_dryness, 
        args.reverb_damping
    )
    
    # Apply pitch change to instrumental if needed
    final_inst_path = str(inst_file)
    if args.pitch_change_all != 0:
        print('[~] Applying overall pitch change to instrumental...')
        import sox
        import soundfile as sf
        
        pitched_inst_path = song_dir / f'instrumental_p{args.pitch_change_all}.wav'
        y, sr = sf.read(str(inst_file))
        tfm = sox.Transformer()
        tfm.pitch(args.pitch_change_all)
        y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
        sf.write(str(pitched_inst_path), y_shifted, sr)
        final_inst_path = str(pitched_inst_path)
    
    # Combine audio
    print('[~] Combining AI Vocals and Instrumentals...')
    final_output_path = song_dir / f'cover_{args.rvc_dirname}.{args.output_format}'
    combine_audio(
        [ai_vocals_mixed_path, final_inst_path], 
        str(final_output_path), 
        args.main_vol, 
        args.inst_vol, 
        args.output_format
    )
    
    print(f'[+] Cover generated at {final_output_path}')
    return str(final_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI cover from separated voice and instrumental files.")
    
    # Input files
    parser.add_argument("--voice_input", type=str, default='tmp/Vocals_No_Noise.wav', help="Path to the voice/vocals audio file")
    parser.add_argument("--inst_input", type=str, default='tmp/Instrumental.wav', help="Path to the instrumental audio file")
    
    # Voice model
    parser.add_argument("--rvc_dirname", type=str, default='Jimin', help="Name of the folder in the rvc_models directory containing the RVC model file and optional index file to use")
    
    # Voice conversion parameters
    parser.add_argument("--pitch_change", type=int, default=0, help="Change the pitch of AI Vocals only. Generally, use 1 for male to female and -1 for vice-versa. (Octaves)")
    parser.add_argument("--pitch_change_all", type=int, default=0, help="Change the pitch/key of vocals and instrumentals. Changing this slightly reduces sound quality")
    parser.add_argument("--index_rate", type=float, default=0.75, help="A decimal number e.g. 0.5, used to reduce/resolve the timbre leakage problem. If set to 1, more biased towards the timbre quality of the training dataset")
    parser.add_argument("--filter_radius", type=int, default=3, help="A number between 0 and 7. If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.")
    parser.add_argument("--pitch_detection_algo", choices=["rmvpe", "mangio-crepe"], default="rmvpe", help="Best option is rmvpe (clarity in vocals), then mangio-crepe (smoother vocals).")
    parser.add_argument("--crepe_hop_length", type=int, default=128, help="If pitch detection algo is mangio-crepe, controls how often it checks for pitch changes in milliseconds. The higher the value, the faster the conversion and less risk of voice cracks, but there is less pitch accuracy. Recommended: 128.")
    parser.add_argument("--protect", type=float, default=0.33, help="A decimal number e.g. 0.33. Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy.")
    parser.add_argument("--remix_mix_rate", type=float, default=0.25, help="A decimal number e.g. 0.25. Control how much to use the original vocal's loudness (0) or a fixed loudness (1).")
    
    # Audio mixing parameters
    parser.add_argument("--main_vol", type=int, default=0, help="Volume change for AI main vocals in decibels. Use -3 to decrease by 3 decibels and 3 to increase by 3 decibels")
    parser.add_argument("--inst_vol", type=int, default=0, help="Volume change for instrumentals in decibels")
    
    # Reverb parameters
    parser.add_argument("--reverb_size", type=float, default=0.15, help="Reverb room size between 0 and 1")
    parser.add_argument("--reverb_wetness", type=float, default=0.2, help="Reverb wet level between 0 and 1")
    parser.add_argument("--reverb_dryness", type=float, default=0.8, help="Reverb dry level between 0 and 1")
    parser.add_argument("--reverb_damping", type=float, default=0.7, help="Reverb damping between 0 and 1")
    
    # Output format
    parser.add_argument("--output_format", choices=["mp3", "wav"], default="mp3", help="Output format of audio file. mp3 for smaller file size, wav for best quality")
    
    args = parser.parse_args()
    
    try:
        generate_ai_cover_from_separated_files(args)
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        sys.exit(1)
