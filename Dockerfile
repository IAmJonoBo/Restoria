FROM python:3.14-slim

# System deps for OpenCV and runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (leverage layer cache)
COPY pyproject.toml README.md LICENSE /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir '.[torch2]' gradio && \
    pip uninstall -y basicsr || true && \
    pip install --no-cache-dir --upgrade 'git+https://github.com/xinntao/BasicSR@master'

# Copy the rest of the project
COPY . /app

EXPOSE 7860

CMD ["gfpgan-gradio", "--server-name", "0.0.0.0", "--server-port", "7860"]

