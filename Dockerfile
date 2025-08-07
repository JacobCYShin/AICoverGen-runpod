# RunPod Serverless용 Audio Separator Dockerfile
FROM runpod/base:0.6.2-cuda12.1.0

# 시스템 패키지 업데이트 및 설치
RUN apt-get update && apt-get upgrade -y


# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 업데이트 (안정적인 버전 사용)
RUN python3 -m pip install --upgrade pip==23.3.1 setuptools wheel

# 작업 디렉토리 설정
WORKDIR /workspace

# 필요한 파일들을 컨테이너에 복사
COPY . /workspace/

# Python 의존성 설치 (단계별로 설치)
RUN pip install numpy==1.24.3 scipy librosa soundfile
RUN pip install ffmpeg-python praat-parselmouth pedalboard pydub pyworld sox
RUN pip install faiss-cpu onnxruntime-gpu
RUN pip install gradio tqdm requests lib
RUN pip install yt_dlp
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install torchcrepe
RUN pip install omegaconf==2.0.6 hydra-core==1.0.7
RUN pip install fairseq==0.12.2

# RVC 기본 모델 다운로드
RUN python3 download_base_models.py

# 핸들러 실행
CMD ["python3", "handler.py"]
