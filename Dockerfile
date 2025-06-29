# ========================
# Base image: Python + CUDA (if using GPU)
# Use cpu only if you don't need GPU
# ========================
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    ffmpeg libx264-dev\
    && rm -rf /var/lib/apt/lists/*



# Copy requirements first for caching
COPY app/requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY app /app

# Copy models (optionally) if needed inside container
COPY models /models

# Expose default streamlit port
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "app.py"]
