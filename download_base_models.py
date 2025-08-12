#!/usr/bin/env python3
"""
RVC 및 MDX 기본 모델 파일 다운로드 스크립트
src/download_models.py를 참고하여 작성
"""

import os
import requests
import hashlib
import time
from pathlib import Path
from urllib.parse import urljoin

# 다운로드 링크
MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
RVC_DOWNLOAD_LINK = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/'

# 파일 크기 검증 (MB 단위)
EXPECTED_FILE_SIZES = {
    'hubert_base.pt': 95.0,  # 약 95MB
    'rmvpe.pt': 15.0,        # 약 15MB
    'UVR-MDX-NET-Voc_FT.onnx': 200.0,  # 약 200MB
    'UVR_MDXNET_KARA_2.onnx': 200.0,   # 약 200MB
    'Reverb_HQ_By_FoxJoy.onnx': 200.0, # 약 200MB
}

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

def verify_file_integrity(file_path, model_name):
    """파일 무결성 검증"""
    if not file_path.exists():
        return False
    
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    expected_size = EXPECTED_FILE_SIZES.get(model_name, 0)
    
    if expected_size > 0:
        # 파일 크기가 예상 크기의 80% 이상이어야 함
        if file_size_mb < expected_size * 0.8:
            print(f"⚠️ {model_name} 파일 크기가 너무 작음: {file_size_mb:.1f}MB (예상: {expected_size}MB)")
            return False
    
    # 파일이 읽을 수 있는지 확인
    try:
        with open(file_path, 'rb') as f:
            # 파일의 처음 1KB를 읽어서 파일이 손상되지 않았는지 확인
            header = f.read(1024)
            if len(header) == 0:
                print(f"⚠️ {model_name} 파일이 비어있음")
                return False
    except Exception as e:
        print(f"⚠️ {model_name} 파일 읽기 실패: {e}")
        return False
    
    return True

def dl_model(link, model_name, dir_name, max_retries=3):
    """모델 파일 다운로드 (재시도 로직 포함)"""
    file_path = dir_name / model_name
    
    # 이미 파일이 있고 무결성이 확인되면 스킵
    if file_path.exists() and verify_file_integrity(file_path, model_name):
        print(f"✅ {model_name} 이미 존재하고 무결성 확인됨")
        return True
    
    # 손상된 파일이 있으면 삭제
    if file_path.exists():
        print(f"🗑️ 손상된 {model_name} 파일 삭제 중...")
        file_path.unlink()
    
    for attempt in range(max_retries):
        print(f'다운로드 중: {model_name} (시도 {attempt + 1}/{max_retries})...')
        try:
            url = urljoin(link, model_name)
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            # 파일 크기 확인
            total_size = int(response.headers.get('content-length', 0))
            if total_size > 0:
                print(f"  예상 크기: {total_size / (1024*1024):.1f}MB")
            
            with open(file_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024*1024) < 8192:  # 1MB마다 진행률 출력
                                print(f"  진행률: {progress:.1f}%")
            
            # 다운로드 후 무결성 검증
            if verify_file_integrity(file_path, model_name):
                print(f'✅ {model_name} 다운로드 완료')
                return True
            else:
                print(f'❌ {model_name} 무결성 검증 실패')
                file_path.unlink()  # 손상된 파일 삭제
                
        except Exception as e:
            print(f'❌ {model_name} 다운로드 실패 (시도 {attempt + 1}): {e}')
            if file_path.exists():
                file_path.unlink()  # 부분적으로 다운로드된 파일 삭제
            
            if attempt < max_retries - 1:
                print(f"  {5}초 후 재시도...")
                time.sleep(5)
    
    print(f'❌ {model_name} 다운로드 최종 실패')
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
