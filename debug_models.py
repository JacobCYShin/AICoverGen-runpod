#!/usr/bin/env python3
"""
Debug script to check model files status
"""

import requests
import json
import time

class RunPodClient:
    def __init__(self, endpoint_id, api_key):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        
    def check_model_files(self):
        """Check the status of model files"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": {
                "operation": "check_model_files"
            }
        }
        
        print("🔍 모델 파일 상태 확인 중...")
        response = requests.post(f"{self.base_url}/runsync", headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"원본 응답: {result}")
            
            if "output" in result:
                output = result["output"]
                print("\n📁 모델 파일 상태:")
                
                # RVC 기본 모델들
                if "rvc_base_models" in output:
                    print("\n🔧 RVC 기본 모델들:")
                    for model_name, info in output["rvc_base_models"].items():
                        status = "✅" if info["exists"] else "❌"
                        print(f"   {status} {model_name}: {info['size_mb']}MB")
                
                # Voice 모델들
                if "voice_models" in output:
                    print("\n🎤 Voice 모델들:")
                    for model_name, info in output["voice_models"].items():
                        status = "✅" if info["exists"] else "❌"
                        files = info.get("files", [])
                        pth_files = [f for f in files if f.endswith('.pth')]
                        index_files = [f for f in files if f.endswith('.index')]
                        
                        print(f"   {status} {model_name}:")
                        print(f"      📂 경로: {info['path']}")
                        print(f"      📄 전체 파일: {files}")
                        print(f"      🎯 .pth 파일: {pth_files}")
                        print(f"      📊 .index 파일: {index_files}")
                
                # 볼륨 정보
                volume_exists = output.get("runpod_volume_exists", False)
                models_dir = output.get("runpod_rvc_models_dir", "N/A")
                print(f"\n💾 볼륨 정보:")
                print(f"   볼륨 존재: {'✅' if volume_exists else '❌'}")
                print(f"   모델 디렉토리: {models_dir}")
            else:
                print("❌ output 필드를 찾을 수 없습니다")
        else:
            print(f"❌ 요청 실패: {response.status_code}")
            print(f"응답: {response.text}")

def main():
    # RunPod 설정 (실제 값으로 변경 필요)
    ENDPOINT_ID = "6ap14ueajzmjxr"  # 실제 엔드포인트 ID
    API_KEY = "ABC123..."  # 실제 API 키
    
    print("🚀 RunPod 모델 파일 디버깅 시작")
    print("=" * 50)
    
    try:
        client = RunPodClient(ENDPOINT_ID, API_KEY)
        client.check_model_files()
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
