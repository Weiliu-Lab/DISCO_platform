FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=60 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com

WORKDIR /appcode

# Debian apt 换阿里云源 + 基础编译依赖
RUN set -eux; \
  if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
    sed -i \
      -e 's|http://deb.debian.org/debian|http://mirrors.aliyun.com/debian|g' \
      -e 's|http://security.debian.org/debian-security|http://mirrors.aliyun.com/debian-security|g' \
      /etc/apt/sources.list.d/debian.sources; \
  fi; \
  if [ -f /etc/apt/sources.list ]; then \
    sed -i \
      -e 's|http://deb.debian.org/debian|http://mirrors.aliyun.com/debian|g' \
      -e 's|http://security.debian.org/debian-security|http://mirrors.aliyun.com/debian-security|g' \
      /etc/apt/sources.list; \
  fi; \
  apt-get update -o Acquire::Retries=5; \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    libffi-dev libssl-dev \
    git curl ca-certificates \
  ; \
  rm -rf /var/lib/apt/lists/*

# venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 依赖
COPY requirements-ui.txt /tmp/requirements-ui.txt
COPY backend/requirements.txt /tmp/backend-requirements.txt

RUN pip install -U pip setuptools wheel \
 && pip install -r /tmp/requirements-ui.txt \
 && pip install -r /tmp/backend-requirements.txt \
 # 兜底：如果 requirements 里漏了 Django，就补装（否则你就会看到你现在的错误）
 && (pip show Django >/dev/null 2>&1 || pip install "Django>=4.2,<6")

# 代码
COPY . /appcode

EXPOSE 5000

