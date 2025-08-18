# RunPod Serverless용 AICoverGen Dockerfile
FROM runpod/base:0.6.2-cuda12.1.0

# 시스템 패키지 업데이트 및 설치
RUN apt-get update && apt-get upgrade -y

# Python 명시적 설치
RUN apt-get install -y python3 python3-pip python3-venv

# 필수 패키지 설치
RUN apt-get install -y \
    ffmpeg \
    cuda-toolkit \
    cudnn9-cuda-12 \
    wget \
    curl \
    git \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 업데이트
RUN python3 -m pip install --upgrade pip==23.3.1 setuptools wheel

# Python 심볼릭 링크 생성 (python 명령어도 사용 가능하도록)
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Python 경로 확인
RUN which python3 && python3 --version

# CUDA 12 호환 ONNX Runtime 설치
RUN pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# 작업 디렉토리 설정
WORKDIR /workspace

# 필요한 파일들을 컨테이너에 복사
COPY . /workspace/

# Python 의존성 설치 (단계별로 설치)
RUN pip install numpy==1.24.3 scipy librosa soundfile
RUN pip install ffmpeg-python praat-parselmouth pedalboard pydub pyworld sox
RUN pip install faiss-cpu onnxruntime-gpu
RUN pip install gradio tqdm requests lib tensorboardX
RUN pip install yt_dlp
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install torchcrepe
RUN pip install omegaconf==2.0.6 hydra-core==1.0.7
RUN pip install fairseq==0.12.2
RUN pip install runpod
RUN pip install boto3==1.34.0 python-dotenv==1.0.0

# 모델 캐시 디렉토리 생성
RUN mkdir -p /runpod-volume/rvc_models
RUN mkdir -p /runpod-volume/mdxnet_models
RUN mkdir -p /runpod-volume/output

# RVC 기본 모델 다운로드 (안정성 개선)
RUN echo "모델 다운로드 시작..." && \
    python3 download_base_models.py && \
    echo "모델 다운로드 완료" && \
    echo "다운로드된 파일 확인:" && \
    ls -la /runpod-volume/rvc_models/ && \
    ls -la /runpod-volume/mdxnet_models/

# 런포드/앱 환경 변수 기본값
ENV RP_HANDLER_TIMEOUT=900 \
    RP_UPLOAD_ENABLE=true \
    RP_VERBOSE=true \
    PRELOAD_MODELS=false \
    OUTPUT_DIR=/runpod-volume/output/

# 헬스체크 (서버가 시작되었는지 간단 확인)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD pgrep -f "handler.py" || exit 1

# 포트 노출
EXPOSE 8000

# 핸들러 실행
CMD ["python3", "handler.py"]
