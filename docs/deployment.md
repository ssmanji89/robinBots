# Deployment Guide

## Prerequisites

- Docker installed on the deployment machine
- Access to a Docker registry (e.g., Docker Hub)
- SSH access to the deployment server
- Necessary environment variables and configuration files

## Building the Docker Image

1. Navigate to the project root directory.

2. Build the Docker image:
   ```
   docker build -t robinbots:latest .
   ```

3. Tag the image for your registry:
   ```
   docker tag robinbots:latest your-registry/robinbots:latest
   ```

4. Push the image to your registry:
   ```
   docker push your-registry/robinbots:latest
   ```

## Preparing the Deployment Server

1. SSH into your server:
   ```
   ssh user@your-server-ip
   ```

2. Install Docker if not already installed:
   ```
   sudo apt-get update
   sudo apt-get install docker.io
   ```

3. Create a directory for the application:
   ```
   mkdir -p /opt/robinbots
   cd /opt/robinbots
   ```

4. Create a `.env` file with necessary environment variables:
   ```
   touch .env
   nano .env
   # Add your environment variables here
   ```

## Deploying the Application

1. Pull the latest image:
   ```
   docker pull your-registry/robinbots:latest
   ```

2. Stop and remove the existing container (if any):
   ```
   docker stop robinbots || true
   docker rm robinbots || true
   ```

3. Run the new container:
   ```
   docker run -d --name robinbots      --env-file /opt/robinbots/.env      -v /opt/robinbots/data:/app/data      -v /opt/robinbots/logs:/app/logs      --restart unless-stopped      your-registry/robinbots:latest
   ```

## Monitoring the Deployment

1. Check if the container is running:
   ```
   docker ps
   ```

2. View the logs:
   ```
   docker logs -f robinbots
   ```

3. Monitor system resources:
   ```
   docker stats robinbots
   ```

## Updating the Application

1. Pull the latest image:
   ```
   docker pull your-registry/robinbots:latest
   ```

2. Stop and remove the existing container:
   ```
   docker stop robinbots
   docker rm robinbots
   ```

3. Run the new container (same command as in the Deploying section).

## Backup and Restore

1. Backup the data directory:
   ```
   tar -czvf robinbots_data_backup.tar.gz /opt/robinbots/data
   ```

2. To restore:
   ```
   tar -xzvf robinbots_data_backup.tar.gz -C /
   ```

## Troubleshooting

- If the container fails to start, check the logs:
  ```
  docker logs robinbots
  ```

- Ensure all required environment variables are set in the `.env` file.

- Check system resources (CPU, memory, disk space) to ensure they're not exhausted.

Remember to always test the deployment process in a staging environment before applying it to production. Regularly review and update your deployment process to incorporate best practices and new features.
