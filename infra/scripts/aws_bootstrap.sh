#!/usr/bin/env bash
#
# One-time AWS bootstrap for the EC2 + Docker Compose + GitHub Actions
# deployment path. Plain AWS CLI, no Terraform — matches this project's
# existing style (tasks.py, preflight.py): a script you can read top to
# bottom, not a state file you have to trust.
#
# Idempotent: safe to re-run. Each resource is looked up before being
# created, so a partial prior run does not produce duplicates.
#
# What this creates (all free or near-free — the ONE billable resource is
# the EC2 instance, launched by a separate step, not this script):
#   - two ECR repositories: npn-api, npn-frontend
#   - an IAM OIDC identity provider for GitHub Actions (if none exists)
#   - an IAM role GitHub Actions assumes via OIDC, scoped to ONE repo and
#     ONE branch — never a static AWS key stored in GitHub
#   - an IAM role + instance profile for the EC2 host itself (SSM + ECR
#     pull only — no S3, no admin, nothing broader than it needs)
#   - a security group with a single inbound rule: tcp/8080 from anywhere
#     (the app's public port under --prod). No inbound 22 — the deploy
#     path uses SSM Session Manager, never SSH.
#
# What this does NOT do: launch the EC2 instance, or touch anything in
# your GitHub repo. Both are separate, deliberately reviewable steps.
#
# Usage:
#   GITHUB_REPO=org/name ./aws_bootstrap.sh
#
set -euo pipefail

: "${GITHUB_REPO:?Set GITHUB_REPO=owner/repo before running this script}"
: "${AWS_REGION:=$(aws configure get region)}"
: "${PROJECT:=npn-hackathon}"

echo "== AWS bootstrap for ${PROJECT} =="
echo "   region:       ${AWS_REGION}"
echo "   github repo:  ${GITHUB_REPO}"
echo

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# ---------------------------------------------------------------------------
# 1. ECR repositories
# ---------------------------------------------------------------------------
for repo in npn-api npn-frontend; do
    if aws ecr describe-repositories --repository-names "$repo" >/dev/null 2>&1; then
        echo "[ ok ] ECR repo already exists: $repo"
    else
        aws ecr create-repository \
            --repository-name "$repo" \
            --image-scanning-configuration scanOnPush=true \
            --tags Key=Project,Value="$PROJECT" >/dev/null
        echo "[ + ]  created ECR repo: $repo"
    fi
done
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# ---------------------------------------------------------------------------
# 2. Data-layer S3 bucket — one-time transfer of the 130 MB product data
#    layer to the host, and the deploy-time handoff point for the compose
#    file (uploaded under compose/ by the `publish` GitHub Actions job).
#    Not a runtime dependency of the app itself — see docker-compose.deploy.yml.
# ---------------------------------------------------------------------------
DATA_BUCKET="${PROJECT}-data-${ACCOUNT_ID}"
if aws s3api head-bucket --bucket "$DATA_BUCKET" >/dev/null 2>&1; then
    echo "[ ok ] data bucket already exists: $DATA_BUCKET"
else
    aws s3api create-bucket --bucket "$DATA_BUCKET" --region "$AWS_REGION" >/dev/null
    aws s3api put-public-access-block --bucket "$DATA_BUCKET" --public-access-block-configuration \
        BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    aws s3api put-bucket-tagging --bucket "$DATA_BUCKET" \
        --tagging "TagSet=[{Key=Project,Value=${PROJECT}}]"
    echo "[ + ]  created data bucket: $DATA_BUCKET (all public access blocked)"
    echo "       upload the data layer once with:"
    echo "       aws s3 cp backend/data/ s3://${DATA_BUCKET}/ --recursive --exclude '*' --include '*.duckdb' --include '*.parquet'"
fi

# ---------------------------------------------------------------------------
# 3. GitHub OIDC identity provider (one per AWS account, shared across repos)
# ---------------------------------------------------------------------------
OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"
if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" >/dev/null 2>&1; then
    echo "[ ok ] GitHub OIDC provider already exists"
else
    aws iam create-open-id-connect-provider \
        --url "https://token.actions.githubusercontent.com" \
        --client-id-list "sts.amazonaws.com" \
        --thumbprint-list "6938fd4d98bab03faadb97b34396831e3780aea1" \
        --tags Key=Project,Value="$PROJECT" >/dev/null
    echo "[ + ]  created GitHub OIDC provider"
fi

# ---------------------------------------------------------------------------
# 4. IAM role GitHub Actions assumes — scoped to this repo, main branch only
# ---------------------------------------------------------------------------
GHA_ROLE_NAME="${PROJECT}-github-deploy"
TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Federated": "${OIDC_ARN}"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {"token.actions.githubusercontent.com:aud": "sts.amazonaws.com"},
      "StringLike": {"token.actions.githubusercontent.com:sub": "repo:${GITHUB_REPO}:ref:refs/heads/main"}
    }
  }]
}
EOF
)

if aws iam get-role --role-name "$GHA_ROLE_NAME" >/dev/null 2>&1; then
    aws iam update-assume-role-policy --role-name "$GHA_ROLE_NAME" \
        --policy-document "$TRUST_POLICY" >/dev/null
    echo "[ ok ] IAM role exists, trust policy refreshed: $GHA_ROLE_NAME"
else
    aws iam create-role --role-name "$GHA_ROLE_NAME" \
        --assume-role-policy-document "$TRUST_POLICY" \
        --tags Key=Project,Value="$PROJECT" >/dev/null
    echo "[ + ]  created IAM role: $GHA_ROLE_NAME"
fi

# Least-privilege inline policy: push to these two repos, and run SSM
# commands only against instances tagged for this project — not "*".
GHA_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "EcrAuth",
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
      "Sid": "EcrPush",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability", "ecr:PutImage",
        "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload", "ecr:BatchGetImage"
      ],
      "Resource": [
        "arn:aws:ecr:${AWS_REGION}:${ACCOUNT_ID}:repository/npn-api",
        "arn:aws:ecr:${AWS_REGION}:${ACCOUNT_ID}:repository/npn-frontend"
      ]
    },
    {
      "Sid": "SsmDocument",
      "Effect": "Allow",
      "Action": "ssm:SendCommand",
      "Resource": "arn:aws:ssm:${AWS_REGION}::document/AWS-RunShellScript"
    },
    {
      "Sid": "SsmTargetInstance",
      "Effect": "Allow",
      "Action": "ssm:SendCommand",
      "Resource": "arn:aws:ec2:${AWS_REGION}:${ACCOUNT_ID}:instance/*",
      "Condition": {
        "StringEquals": {"ssm:resourceTag/Project": "${PROJECT}"}
      }
    },
    {
      "Sid": "SsmReadResult",
      "Effect": "Allow",
      "Action": "ssm:GetCommandInvocation",
      "Resource": "*"
    },
    {
      "Sid": "PublishComposeFile",
      "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": "arn:aws:s3:::${DATA_BUCKET}/compose/*"
    }
  ]
}
EOF
)
aws iam put-role-policy --role-name "$GHA_ROLE_NAME" \
    --policy-name "${PROJECT}-deploy-policy" \
    --policy-document "$GHA_POLICY" >/dev/null
echo "[ ok ] deploy policy attached to $GHA_ROLE_NAME"
GHA_ROLE_ARN=$(aws iam get-role --role-name "$GHA_ROLE_NAME" --query 'Role.Arn' --output text)

# ---------------------------------------------------------------------------
# 5. IAM role + instance profile for the EC2 host — SSM + ECR pull only
# ---------------------------------------------------------------------------
EC2_ROLE_NAME="${PROJECT}-ec2-role"
EC2_TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"},
                  "Action": "sts:AssumeRole"}]
}'
if aws iam get-role --role-name "$EC2_ROLE_NAME" >/dev/null 2>&1; then
    echo "[ ok ] EC2 IAM role already exists: $EC2_ROLE_NAME"
else
    aws iam create-role --role-name "$EC2_ROLE_NAME" \
        --assume-role-policy-document "$EC2_TRUST_POLICY" \
        --tags Key=Project,Value="$PROJECT" >/dev/null
    echo "[ + ]  created EC2 IAM role: $EC2_ROLE_NAME"
fi
for policy in AmazonSSMManagedInstanceCore AmazonEC2ContainerRegistryReadOnly; do
    aws iam attach-role-policy --role-name "$EC2_ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/${policy}" 2>/dev/null || true
done
echo "[ ok ] SSM + ECR-readonly policies attached to $EC2_ROLE_NAME"

# Inline: read-only access to exactly the data bucket (for the one-time data
# transfer and the compose-file handoff) and exactly the Gemini SSM
# parameter (fetched at deploy time to populate the container's environment
# — never written to disk, never in an image layer).
EC2_INLINE_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadDataBucket",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::${DATA_BUCKET}", "arn:aws:s3:::${DATA_BUCKET}/*"]
    },
    {
      "Sid": "ReadGeminiKey",
      "Effect": "Allow",
      "Action": "ssm:GetParameter",
      "Resource": "arn:aws:ssm:${AWS_REGION}:${ACCOUNT_ID}:parameter/${PROJECT}/gemini-api-key"
    },
    {
      "Sid": "DecryptGeminiKey",
      "Effect": "Allow",
      "Action": "kms:Decrypt",
      "Resource": "arn:aws:kms:${AWS_REGION}:${ACCOUNT_ID}:alias/aws/ssm"
    }
  ]
}
EOF
)
aws iam put-role-policy --role-name "$EC2_ROLE_NAME" \
    --policy-name "${PROJECT}-ec2-inline-policy" \
    --policy-document "$EC2_INLINE_POLICY" >/dev/null
echo "[ ok ] data-bucket-read + gemini-key-read policy attached to $EC2_ROLE_NAME"

INSTANCE_PROFILE="${PROJECT}-ec2-profile"
if aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" >/dev/null 2>&1; then
    echo "[ ok ] instance profile already exists: $INSTANCE_PROFILE"
else
    aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE" >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$INSTANCE_PROFILE" \
        --role-name "$EC2_ROLE_NAME" >/dev/null
    echo "[ + ]  created instance profile: $INSTANCE_PROFILE"
    echo "       (IAM propagation can take ~10s before EC2 will accept it)"
fi

# ---------------------------------------------------------------------------
# 6. Security group — inbound 8080 only, no SSH
# ---------------------------------------------------------------------------
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" \
    --query 'Vpcs[0].VpcId' --output text)
SG_NAME="${PROJECT}-sg"
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=${SG_NAME}" "Name=vpc-id,Values=${VPC_ID}" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    SG_ID=$(aws ec2 create-security-group --group-name "$SG_NAME" \
        --description "npn-hackathon app: inbound 8080 only, no SSH (SSM only)" \
        --vpc-id "$VPC_ID" \
        --tag-specifications "ResourceType=security-group,Tags=[{Key=Project,Value=${PROJECT}}]" \
        --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
        --protocol tcp --port 8080 --cidr 0.0.0.0/0 >/dev/null
    echo "[ + ]  created security group: $SG_ID (tcp/8080 open, no port 22)"
else
    echo "[ ok ] security group already exists: $SG_ID"
fi

# ---------------------------------------------------------------------------
# 7. GEMINI_API_KEY placeholder in SSM Parameter Store (SecureString)
# ---------------------------------------------------------------------------
PARAM_NAME="/${PROJECT}/gemini-api-key"
if aws ssm get-parameter --name "$PARAM_NAME" >/dev/null 2>&1; then
    echo "[ ok ] SSM parameter already exists: $PARAM_NAME (value untouched)"
else
    aws ssm put-parameter --name "$PARAM_NAME" --type SecureString --value "unset" \
        --tags Key=Project,Value="$PROJECT" >/dev/null
    echo "[ + ]  created SSM parameter: $PARAM_NAME (value: \"unset\" — update with"
    echo "       aws ssm put-parameter --name $PARAM_NAME --type SecureString --overwrite --value <key>)"
fi

# ---------------------------------------------------------------------------
# Summary — feed these into the GitHub Actions workflow / EC2 launch step
# ---------------------------------------------------------------------------
echo
echo "== done =="
echo "ECR_REGISTRY        = ${ECR_REGISTRY}"
echo "GHA_ROLE_ARN         = ${GHA_ROLE_ARN}"
echo "EC2_INSTANCE_PROFILE = ${INSTANCE_PROFILE}"
echo "SECURITY_GROUP_ID    = ${SG_ID}"
echo "VPC_ID               = ${VPC_ID}"
