#!/bin/bash
# deploy.sh

echo "🚀 Deploying AI Script Summarizer..."

# Create necessary directories
mkdir -p logs ssl

# Set permissions
chmod +x deploy.sh

# Build and start services
echo "📦 Building Docker images..."
docker-compose build

echo "🔄 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check health
echo "🔍 Checking service health..."
curl -f http://localhost/health || echo "❌ Health check failed"

echo "✅ Deployment completed!"
echo "📖 API Documentation: http://localhost/docs"
echo "📊 Health Check: http://localhost/health"
echo "🔧 API Endpoints:"
echo "  - POST /api/summarize - Tạo tóm tắt"
echo "  - POST /api/train - Huấn luyện model"
echo "  - GET /api/stats - Thống kê"

