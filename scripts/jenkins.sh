#!/bin/bash

# Update system packages
sudo apt update 

# Install Java (Jenkins requires Java to run)
sudo apt install openjdk-8-jdk -y

# Add Jenkins repository and install Jenkins
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io.key | sudo tee /usr/share/keyrings/jenkins-keyring.asc > /dev/null
echo deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] https://pkg.jenkins.io/debian-stable binary/ | sudo tee /etc/apt/sources.list.d/jenkins.list > /dev/null

# Update package list and install Jenkins
sudo apt update
sudo apt install jenkins -y

# Start and enable Jenkins service
sudo systemctl start jenkins
sudo systemctl enable jenkins

# Check Jenkins service status
sudo systemctl status jenkins

## Installing Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add current user and Jenkins to the Docker group
sudo usermod -aG docker $USER
sudo usermod -aG docker jenkins

# Reload group memberships
newgrp docker

# Install AWS CLI
sudo apt install awscli -y

# Re-add Jenkins to Docker group (this step may not be necessary if already added above)
sudo usermod -a -G docker jenkins

# Configure AWS credentials (ensure you have your credentials handy)
aws configure

# Display Jenkins initial admin password for setup
echo "To complete Jenkins setup, use the following admin password:"
sudo cat /var/lib/jenkins/secrets/initialAdminPassword

# You can now access Jenkins at http://your-elastic-ip:8080
