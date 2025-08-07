#!/usr/bin/env python3
"""
RunPod Serverless용 모델 파일 업로드 스크립트
"""

import os
import subprocess
import sys
from pathlib import Path

def upload_to_runpod(local_path: str, runpod_path: str):
    """RunPod에 파일/폴더 업로드"""
    try:
        print(f"[~] Uploading {local_path} to {runpod_path}...")
        
        # RunPod CLI를 사용해서 업로드
        cmd = ["runpod", "upload", local_path, runpod_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[+] Successfully uploaded {local_path}")
            return True
        else:
            print(f"[-] Failed to upload {local_path}: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[-] Error uploading {local_path}: {str(e)}")
        return False

def main():
    """메인 함수"""
    print("=== RunPod Serverless 모델 업로드 ===")
    
    # 현재 디렉토리 확인
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # 업로드할 모델들
    models_to_upload = [
        ("rvc_models", "/runpod-volume/rvc_models"),
        ("mdxnet_models", "/runpod-volume/mdxnet_models"),
    ]
    
    success_count = 0
    total_count = len(models_to_upload)
    
    for local_path, runpod_path in models_to_upload:
        if os.path.exists(local_path):
            if upload_to_runpod(local_path, runpod_path):
                success_count += 1
        else:
            print(f"[-] Local path {local_path} does not exist, skipping...")
    
    print(f"\n=== 업로드 완료 ===")
    print(f"성공: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("✅ 모든 모델이 성공적으로 업로드되었습니다!")
        print("\n다음 단계:")
        print("1. runpod deploy --config runpod.yaml")
        print("2. 배포된 엔드포인트로 테스트")
    else:
        print("❌ 일부 모델 업로드에 실패했습니다.")
        print("로컬 모델 파일들을 확인해주세요.")

if __name__ == "__main__":
    main()
