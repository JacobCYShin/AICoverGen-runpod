#!/usr/bin/env python3
"""
RVC 및 MDX 기본 모델 파일 다운로드 스크립트
원본 src/download_models.py를 기반으로 /runpod-volume 지원 추가
"""

import os
import requests
from pathlib import Path

# 다운로드 링크 (원본과 동일)
MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
RVC_DOWNLOAD_LINK = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/'

# 디렉토리 설정
BASE_DIR = Path(__file__).resolve().parent

# RunPod Serverless 환경에서는 /runpod-volume 사용, 로컬에서는 /workspace 사용
if os.path.exists('/runpod-volume'):
    # RunPod Serverless 환경
    mdxnet_models_dir = Path('/runpod-volume/mdxnet_models')
    rvc_models_dir = Path('/runpod-volume/rvc_models')
    print("🔧 RunPod Serverless 환경 감지: /runpod-volume 사용")
else:
    # 로컬 환경
    mdxnet_models_dir = BASE_DIR / 'mdxnet_models'
    rvc_models_dir = BASE_DIR / 'rvc_models'
    print("🔧 로컬 환경 감지: /workspace 사용")

def dl_model(link, model_name, dir_name):
    """모델 파일 다운로드 (원본과 동일한 간단한 구조)"""
    print(f'다운로드 중: {model_name}...')
    try:
        with requests.get(f'{link}{model_name}') as r:
            r.raise_for_status()
            with open(dir_name / model_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f'✅ {model_name} 다운로드 완료')
        return True
    except Exception as e:
        print(f'❌ {model_name} 다운로드 실패: {e}')
        return False

def main():
    """메인 함수"""
    print("🚀 RVC 및 MDX 기본 모델 다운로드 시작")
    
    # 디렉토리 생성
    mdxnet_models_dir.mkdir(exist_ok=True)
    rvc_models_dir.mkdir(exist_ok=True)
    
    # MDX 모델 다운로드
    print("\n📥 MDX 모델 다운로드 중...")
    mdx_model_names = ['UVR-MDX-NET-Voc_FT.onnx', 'UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx']
    for model in mdx_model_names:
        if not (mdxnet_models_dir / model).exists():
            dl_model(MDX_DOWNLOAD_LINK, model, mdxnet_models_dir)
        else:
            print(f"✅ {model} 이미 존재함")
    
    # RVC 모델 다운로드
    print("\n📥 RVC 모델 다운로드 중...")
    rvc_model_names = ['hubert_base.pt', 'rmvpe.pt']
    for model in rvc_model_names:
        if not (rvc_models_dir / model).exists():
            dl_model(RVC_DOWNLOAD_LINK, model, rvc_models_dir)
        else:
            print(f"✅ {model} 이미 존재함")
    
    # 다운로드된 파일 목록 출력
    print("\n📋 다운로드된 파일 목록:")
    
    print("  MDX 모델:")
    for model in mdx_model_names:
        file_path = mdxnet_models_dir / model
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"    - {model} ({size:.1f} MB)")
        else:
            print(f"    - {model} (없음)")
    
    print("  RVC 모델:")
    for model in rvc_model_names:
        file_path = rvc_models_dir / model
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"    - {model} ({size:.1f} MB)")
        else:
            print(f"    - {model} (없음)")
    
    print("\n🎉 모든 모델 다운로드 완료!")

if __name__ == "__main__":
    main()
