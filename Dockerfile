FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user first
RUN useradd -m -u 1000 appuser

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create NLTK data directory
RUN mkdir -p /home/appuser/nltk_data

# Download NLTK data as non-root user
RUN python -c "import nltk; nltk.data.path.append('/home/appuser/nltk_data'); nltk.download('punkt', download_dir='/home/appuser/nltk_data'); nltk.download('stopwords', download_dir='/home/appuser/nltk_data')"

# Set NLTK data path
ENV NLTK_DATA=/home/appuser/nltk_data

# Copy application code and change ownership
COPY --chown=appuser:appuser . .

# Create logs directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 7000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7000/api/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7000", "--workers", "1"]
