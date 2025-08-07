#!/usr/bin/env python3
"""
RunPod Serverless AICoverGen 간단 테스트 실행 스크립트
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 테스트 모듈 임포트
from test_runpod_aicovergen import *

def main():
    """메인 실행 함수"""
    print("🎵 RunPod Serverless AICoverGen 테스트 시작")
    print("="*60)
    
    # 설정 확인
    if API_KEY == "YOUR_RUNPOD_API_KEY" or ENDPOINT_ID == "YOUR_ENDPOINT_ID":
        print("❌ API_KEY와 ENDPOINT_ID를 설정해주세요!")
        print("test_runpod_aicovergen.py 파일을 열어서 다음을 수정하세요:")
        print("   - API_KEY = 'your_api_key_here'")
        print("   - ENDPOINT_ID = 'your_endpoint_id_here'")
        return
    
    # 파일 존재 확인
    if not os.path.exists(VOICE_AUDIO_PATH):
        print(f"❌ 보컬 파일을 찾을 수 없습니다: {VOICE_AUDIO_PATH}")
        print("tmp/Vocals_No_Noise.wav 파일이 있는지 확인해주세요.")
        return
    
    if not os.path.exists(INSTRUMENT_AUDIO_PATH):
        print(f"❌ 악기 파일을 찾을 수 없습니다: {INSTRUMENT_AUDIO_PATH}")
        print("tmp/Instrumental.wav 파일이 있는지 확인해주세요.")
        return
    
    # 테스트 실행
    try:
        # 1. 헬스체크
        if not test_health_check():
            print("❌ 서버가 정상 작동하지 않습니다. 종료합니다.")
            return
        
        print("\n" + "="*60)
        
        # 2. 사용 가능한 모델 확인
        available_models = test_list_models()
        if VOICE_MODEL not in available_models:
            print(f"⚠️ 경고: '{VOICE_MODEL}' 모델이 사용 가능한 모델 목록에 없습니다.")
            if available_models:
                print(f"사용 가능한 모델 중 하나를 선택하세요: {available_models}")
                global VOICE_MODEL
                VOICE_MODEL = available_models[0]
                print(f"자동으로 '{VOICE_MODEL}' 모델을 사용합니다.")
        
        print("\n" + "="*60)
        
        # 3. 기본 테스트
        test_generate_cover()
        
        print("\n🎉 테스트 완료!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()
