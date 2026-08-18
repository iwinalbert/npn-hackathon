#!/usr/bin/env bash
#
# Runs ON the EC2 host, invoked by the GitHub Actions `deploy` job over SSM
# (infra/scripts/aws_bootstrap.sh sets up the IAM role and security group
# this depends on). Not something a developer runs directly — for a manual
# or emergency deploy from a workstation with SSH/console access instead,
# see infra/docs/runbook.md, whose steps are the same four operations this
# script automates: log in to ECR, fetch the secret, `compose up`, prune.
#
# Deliberately a plain script, not embedded inline in the GitHub Actions
# YAML — SSM command bodies with heavy quoting are unreadable and untestable.
# This one is neither.
set -euo pipefail

: "${ECR_REGISTRY:?}"
: "${IMAGE_TAG:?}"
: "${AWS_REGION:?}"
: "${GEMINI_PARAM_NAME:?}"

echo "== deploying ${IMAGE_TAG} =="

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

# The placeholder value written by aws_bootstrap.sh means "no key configured
# yet" — pass that through as empty so the assistant reports unavailable
# instead of trying to authenticate with the literal string "unset".
GEMINI_API_KEY=$(aws ssm get-parameter --name "$GEMINI_PARAM_NAME" \
  --with-decryption --region "$AWS_REGION" \
  --query Parameter.Value --output text)
if [ "$GEMINI_API_KEY" = "unset" ]; then
  GEMINI_API_KEY=""
fi

cd /opt/npn
ECR_REGISTRY="$ECR_REGISTRY" IMAGE_TAG="$IMAGE_TAG" GEMINI_API_KEY="$GEMINI_API_KEY" \
  docker compose -f docker-compose.deploy.yml up -d

# Old image layers accumulate one deploy at a time; this is what keeps the
# 20 GB root volume from filling on a box that redeploys every push.
docker image prune -f

echo "== deploy command complete =="
docker compose -f docker-compose.deploy.yml ps
