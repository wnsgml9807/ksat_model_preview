#!/bin/bash

# GCP 서비스 계정 생성 및 키 다운로드 스크립트
# KSAT Agent 프로젝트용 자동 설정

PROJECT_ID="gen-lang-client-0921402604"
SERVICE_ACCOUNT_NAME="streamlit-vertex-ai"
KEY_FILE="$HOME/streamlit-vertex-key.json"

echo "🔧 프로젝트 설정: $PROJECT_ID"
gcloud config set project $PROJECT_ID

echo "👤 서비스 계정 생성 중..."
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
    --display-name="Streamlit Vertex AI Service Account" \
    --project=$PROJECT_ID

echo "🔑 권한 부여 중..."
# Vertex AI 엔드포인트 호출 권한 (predict 포함)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# ML 개발 관련 권한  
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/ml.developer"

echo "📁 키 파일 생성 중..."
gcloud iam service-accounts keys create $KEY_FILE \
    --iam-account=$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com

echo "✅ 완료!"
echo "🔑 키 파일 위치: $KEY_FILE"
echo "📋 Streamlit Secrets에 복사할 내용:"
echo "----------------------------------------"
cat $KEY_FILE
echo "----------------------------------------"
echo ""
echo "🚀 Streamlit Cloud > Settings > Secrets에서 다음과 같이 추가하세요:"
echo "GOOGLE_APPLICATION_CREDENTIALS_JSON = '''"
echo "$(cat $KEY_FILE)"
echo "'''"
echo ""
echo "🔗 추가 정보:"
echo "   - 프로젝트 ID: $PROJECT_ID"
echo "   - 엔드포인트 ID: 4075215603537805312"
echo "   - 리전: us-central1"
echo "   - 모델 ID: 6275144856671092736"
