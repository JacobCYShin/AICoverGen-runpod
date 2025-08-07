#!/usr/bin/env python3
"""
AICoverGen Handler 테스트 스크립트
로컬에서 handler.py의 기능을 테스트합니다.
"""

import os
import sys
import json
import base64
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

# handler 모듈 import
from handler import runpod_handler

def create_test_audio(duration=5.0, sample_rate=44100, filename="test_audio.wav"):
    """테스트용 오디오 파일 생성"""
    # 간단한 사인파 생성
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # 440Hz 사인파 (A4 노트)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    # 스테레오로 변환
    audio_stereo = np.column_stack((audio, audio))
    
    # WAV 파일로 저장
    sf.write(filename, audio_stereo, sample_rate)
    return filename

def audio_to_base64(audio_path):
    """오디오 파일을 base64로 인코딩"""
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
    return base64.b64encode(audio_data).decode('utf-8')

def base64_to_audio(base64_data, output_path):
    """base64 데이터를 오디오 파일로 디코딩"""
    audio_data = base64.b64decode(base64_data)
    with open(output_path, 'wb') as f:
        f.write(audio_data)

def test_health_check():
    """Health check 테스트"""
    print("=== Health Check 테스트 ===")
    
    event = {"operation": "health_check"}
    result = runpod_handler(event, None)
    
    print(f"결과: {json.dumps(result, indent=2, ensure_ascii=False)}")
    return result.get("status") == "healthy"

def test_list_models():
    """List models 테스트"""
    print("\n=== List Models 테스트 ===")
    
    event = {"operation": "list_models"}
    result = runpod_handler(event, None)
    
    print(f"결과: {json.dumps(result, indent=2, ensure_ascii=False)}")
    return "models" in result

def test_generate_cover():
    """Generate cover 테스트"""
    print("\n=== Generate Cover 테스트 ===")
    
    # 실제 오디오 파일 사용
    voice_file = "tmp/Vocals_No_Noise.wav"
    instrument_file = "tmp/Instrumental.wav"
    
    if not os.path.exists(voice_file):
        print(f"❌ 보컬 파일이 없습니다: {voice_file}")
        return False
    
    if not os.path.exists(instrument_file):
        print(f"❌ 악기 파일이 없습니다: {instrument_file}")
        return False
    
    print(f"✅ 보컬 파일 확인: {voice_file}")
    print(f"✅ 악기 파일 확인: {instrument_file}")
    
    # 파일 크기 확인
    voice_size = os.path.getsize(voice_file) / (1024 * 1024)  # MB
    instrument_size = os.path.getsize(instrument_file) / (1024 * 1024)  # MB
    print(f"보컬 파일 크기: {voice_size:.1f} MB")
    print(f"악기 파일 크기: {instrument_size:.1f} MB")
    
    # base64로 인코딩
    print("파일을 base64로 인코딩 중...")
    voice_base64 = audio_to_base64(voice_file)
    instrument_base64 = audio_to_base64(instrument_file)
    
    print(f"보컬 base64 길이: {len(voice_base64)}")
    print(f"악기 base64 길이: {len(instrument_base64)}")
    
    # 사용 가능한 모델 확인
    models_result = runpod_handler({"operation": "list_models"}, None)
    available_models = models_result.get("models", [])
    
    if not available_models:
        print("사용 가능한 모델이 없습니다. 테스트를 건너뜁니다.")
        return False
    
    # Jimin 모델 사용
    test_model = "Jimin"
    if test_model not in available_models:
        print(f"❌ {test_model} 모델이 사용 가능한 모델 목록에 없습니다.")
        print(f"사용 가능한 모델: {available_models}")
        return False
    
    print(f"테스트 모델: {test_model}")
    
    # API 호출 (main.py와 동일한 파라미터 사용)
    event = {
        "operation": "generate_cover_from_separate_audio",
        "params": {
            "voice_audio": voice_base64,
            "instrument_audio": instrument_base64,
            "voice_model": test_model,  # Jimin 모델
            "pitch_adjust": 2,  # 피치 조정 (main.py의 pitch_change)
            "index_rate": 0.5,  # 인덱스 비율
            "filter_radius": 3,  # 필터 반지름
            "rms_mix_rate": 0.25,  # RMS 믹스 비율
            "protect": 0.33,  # 보호 비율
            "f0_method": "rmvpe",  # F0 추출 방법
            "crepe_hop_length": 128,  # CREPE 홉 길이
            "pitch_change_all": 0,  # 전체 피치 변경
            "reverb_rm_size": 0.15,  # 리버브 룸 크기
            "reverb_wet": 0.2,  # 리버브 웻 레벨
            "reverb_dry": 0.8,  # 리버브 드라이 레벨
            "reverb_damping": 0.7,  # 리버브 댐핑
            "main_gain": 0,  # 메인 보컬 게인
            "backup_gain": 0,  # 백업 보컬 게인
            "inst_gain": 0,  # 악기 게인
            "output_format": "wav"  # 출력 형식
        }
    }
    
    print("API 호출 시작...")
    result = runpod_handler(event, None)
    
    # 결과 요약만 출력 (오디오 데이터는 제외)
    if isinstance(result, dict):
        result_summary = result.copy()
        if 'audio' in result_summary:
            result_summary['audio'] = f"[BASE64 AUDIO DATA - {len(result_summary['audio'])} bytes]"
        if 'output_audio' in result_summary:
            result_summary['output_audio'] = f"[BASE64 AUDIO DATA - {len(result_summary['output_audio'])} bytes]"
        print(f"API 호출 결과 요약: {result_summary}")
    else:
        print(f"API 호출 결과: {result}")
    
    if result.get("success") == True:
        print("✅ 커버 생성 성공!")
        print(f"오디오 데이터 크기: {len(result.get('output_audio', ''))} bytes")
        
        # 결과 오디오 저장
        output_file = f"test_output_{test_model}.wav"
        base64_to_audio(result["output_audio"], output_file)
        
        # 파일 크기 확인
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"결과 파일 저장: {output_file} (크기: {file_size:.1f} MB)")
        else:
            print(f"결과 파일 저장 실패: {output_file}")
        
        return True
    else:
        print("❌ 커버 생성 실패!")
        print(f"에러: {result.get('error', 'Unknown error')}")
        if 'traceback' in result:
            print("상세 에러 정보:")
            print(result['traceback'])
        else:
            print("전체 결과 객체 (오디오 데이터 제외):")
            result_summary = result.copy()
            if 'audio' in result_summary:
                result_summary['audio'] = f"[BASE64 AUDIO DATA - {len(result_summary['audio'])} bytes]"
            if 'output_audio' in result_summary:
                result_summary['output_audio'] = f"[BASE64 AUDIO DATA - {len(result_summary['output_audio'])} bytes]"
            print(json.dumps(result_summary, indent=2, ensure_ascii=False))
        return False

def test_error_handling():
    """에러 처리 테스트"""
    print("\n=== 에러 처리 테스트 ===")
    
    # 잘못된 operation
    event = {"operation": "invalid_operation"}
    result = runpod_handler(event, None)
    print(f"잘못된 operation 테스트: {result.get('error', 'No error')}")
    
    # 잘못된 파라미터
    event = {
        "operation": "generate_cover_from_separate_audio",
        "params": {
            "voice_audio": "invalid_base64",
            "instrument_audio": "invalid_base64",
            "voice_model": "non_existent_model"
        }
    }
    result = runpod_handler(event, None)
    print(f"잘못된 파라미터 테스트: {result.get('error', 'No error')}")

def cleanup_test_files():
    """테스트 파일 정리"""
    test_files = [
        "test_voice.wav",
        "test_instrument.wav"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"삭제된 파일: {file}")
    
    # 실제 오디오 파일은 삭제하지 않음 (사용자가 제공한 파일이므로)

def main():
    """메인 테스트 함수"""
    print("🚀 AICoverGen Handler 테스트 시작")
    print("📁 실제 오디오 파일 사용: tmp/Vocals_No_Noise.wav, tmp/Instrumental.wav")
    print("🎤 테스트 모델: Jimin")
    print("=" * 50)
    
    try:
        # 테스트 실행
        health_ok = test_health_check()
        models_ok = test_list_models()
        cover_ok = test_generate_cover()
        test_error_handling()
        
        # 결과 요약
        print("\n" + "=" * 50)
        print("📊 테스트 결과 요약")
        print(f"Health Check: {'✅' if health_ok else '❌'}")
        print(f"List Models: {'✅' if models_ok else '❌'}")
        print(f"Generate Cover: {'✅' if cover_ok else '❌'}")
        
        if health_ok and models_ok and cover_ok:
            print("\n🎉 모든 테스트가 성공했습니다!")
        else:
            print("\n⚠️ 일부 테스트가 실패했습니다.")
            
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 테스트 파일 정리
        cleanup_test_files()

if __name__ == "__main__":
    main()
