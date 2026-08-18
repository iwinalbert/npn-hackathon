#!/usr/bin/env bash
set -euo pipefail

: "${ECR_REGISTRY:?}"
: "${IMAGE_TAG:?}"
: "${AWS_REGION:?}"
: "${GEMINI_PARAM_NAME:?}"

echo "== deploying ${IMAGE_TAG} =="

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

GEMINI_API_KEY=$(aws ssm get-parameter --name "$GEMINI_PARAM_NAME" \
  --with-decryption --region "$AWS_REGION" \
  --query Parameter.Value --output text)
if [ "$GEMINI_API_KEY" = "unset" ]; then
  GEMINI_API_KEY=""
fi

cd /opt/npn
ECR_REGISTRY="$ECR_REGISTRY" IMAGE_TAG="$IMAGE_TAG" GEMINI_API_KEY="$GEMINI_API_KEY" \
  docker compose -f docker-compose.deploy.yml up -d

docker image prune -f

echo "== deploy command complete =="
docker compose -f docker-compose.deploy.yml ps
