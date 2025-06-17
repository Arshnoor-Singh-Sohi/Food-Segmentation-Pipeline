#!/usr/bin/env python3
"""
RunPod launcher - automates RunPod setup and deployment
"""

import os
import json
import requests
import subprocess
from pathlib import Path

class RunPodLauncher:
    def __init__(self):
        self.api_key = os.getenv('RUNPOD_API_KEY')
        if not self.api_key:
            print("⚠️  Set RUNPOD_API_KEY environment variable")
        
    def create_pod_config(self):
        """Create optimal pod configuration"""
        return {
            "cloudType": "SECURE",
            "gpuType": "NVIDIA GeForce RTX 3090",
            "gpuCount": 1,
            "containerDiskInGb": 50,
            "volumeInGb": 20,
            "minMemoryInGb": 24,
            "minVcpuCount": 8,
            "dockerArgs": "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root",
            "ports": "8888/http,5000/http",
            "imageName": "runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel",
            "env": [
                {"key": "JUPYTER_TOKEN", "value": "meallens2024"}
            ]
        }
    
    def launch_pod(self):
        """Launch RunPod instance"""
        config = self.create_pod_config()
        
        print("🚀 Launching RunPod with configuration:")
        print(json.dumps(config, indent=2))
        
        # API call would go here
        print("\n📋 Manual steps:")
        print("1. Go to https://runpod.io")
        print("2. Click 'Deploy' -> 'Pods'")
        print("3. Select GPU: RTX 3090")
        print("4. Set container: runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel")
        print("5. Set disk: 50GB")
        print("6. Forward ports: 8888, 5000")
        print("7. Deploy!")
        
        print("\n📦 After deployment:")
        print("1. SSH into pod")
        print("2. Run: git clone <your-repo>")
        print("3. Run: cd food-segmentation-pipeline")
        print("4. Run: bash runpod/start_server.sh")

if __name__ == "__main__":
    launcher = RunPodLauncher()
    launcher.launch_pod()