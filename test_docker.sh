#!/bin/bash

# AICoverGen Docker 테스트 스크립트

echo "🚀 AICoverGen Docker 테스트 시작"
echo "=================================="

# 컨테이너 이름 설정
CONTAINER_NAME="aicovergen-test"
IMAGE_NAME="aicovergen-test"

# 기존 컨테이너가 있으면 삭제
echo "기존 컨테이너 정리 중..."
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Docker 이미지 빌드
echo "Docker 이미지 빌드 중..."
docker build -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo "❌ Docker 빌드 실패!"
    exit 1
fi

echo "✅ Docker 이미지 빌드 완료"

# 컨테이너 실행
echo "컨테이너 실행 중..."
docker run -d --name $CONTAINER_NAME --gpus all $IMAGE_NAME sleep infinity

if [ $? -ne 0 ]; then
    echo "❌ 컨테이너 실행 실패!"
    exit 1
fi

echo "✅ 컨테이너 실행 완료"

# 컨테이너 내부에서 테스트 실행
echo "테스트 실행 중..."
docker exec $CONTAINER_NAME python test_handler.py

# 테스트 결과 확인
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ 테스트 성공!"
else
    echo "❌ 테스트 실패!"
fi

# 컨테이너 정리
echo "컨테이너 정리 중..."
docker rm -f $CONTAINER_NAME

echo "테스트 완료!"
