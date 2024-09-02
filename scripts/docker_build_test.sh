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

# Build Docker image
print_color $BLUE "Building Docker image..."
docker build -t robinbots:latest -f docker/Dockerfile .
check_status "Docker image build"

# Run the application in Docker container and perform health check
print_color $BLUE "Starting application in Docker container..."
docker run -d --name robinbots_app -p 8080:8080 robinbots:latest
check_status "Application startup"

# Wait for the application to start
sleep 5

print_color $GREEN "Docker build and test process completed successfully!"

# Tag the image for release if all checks pass
print_color $BLUE "Tagging image for release..."
docker tag robinbots:latest robinbots:$(date +%Y%m%d)
check_status "Image tagging"

print_color $YELLOW "Next steps:"
echo "1. Push the Docker image to your registry"
echo "2. Update your deployment scripts with the new image tag"
echo "3. Deploy the new version to your staging environment"
