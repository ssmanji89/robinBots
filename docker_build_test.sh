#!/bin/bash

# robinBots Docker Build and Test Script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

# Function to check if a command was successful
check_status() {
    if [ $? -eq 0 ]; then
        print_color $GREEN "✔ $1"
    else
        print_color $RED "✘ $1"
        exit 1
    fi
}

# Cleanup any existing containers and images
print_color $BLUE "Cleaning up old containers and images..."
docker stop robinbots_app 2>/dev/null || true
docker rm robinbots_app 2>/dev/null || true
docker rmi robinbots:latest 2>/dev/null || true
check_status "Cleanup of old containers and images"

# Build Docker image
print_color $BLUE "Building Docker image..."
docker build -t robinbots:latest -f docker/Dockerfile .
check_status "Docker image build"

# Start application in Docker container
print_color $BLUE "Starting application in Docker container..."
docker run -d --name robinbots_app -p 8080:8080 robinbots:latest
check_status "Application startup"

# Wait for the application to start
sleep 5

# Perform health check
print_color $BLUE "Performing health check..."
if curl -f http://localhost:8080/health; then
    print_color $GREEN "Health check passed!"
else
    print_color $RED "Health check failed!"
    docker logs robinbots_app
    docker stop robinbots_app
    docker rm robinbots_app
    exit 1
fi

# Docker container logs (optional)
print_color $BLUE "Application logs:"
docker logs robinbots_app

print_color $GREEN "Docker build and test process completed successfully!"

# Tag the image for release if all checks pass
print_color $BLUE "Tagging image for release..."
IMAGE_TAG="robinbots:$(date +%Y%m%d)"
docker tag robinbots:latest $IMAGE_TAG
check_status "Image tagging"

# Display next steps
print_color $YELLOW "Next steps:"
echo "1. Push the Docker image to your registry: docker push $IMAGE_TAG"
echo "2. Update your deployment scripts with the new image tag"
echo "3. Deploy the new version to your staging environment"

# Optional: Clean up stopped containers (keeps the running one)
print_color $BLUE "Cleaning up stopped containers..."
docker container prune -f
check_status "Stopped containers cleanup"
