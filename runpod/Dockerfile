FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

WORKDIR /workspace

# Copy requirements
COPY requirements_metadata.txt .
RUN pip install -r requirements_metadata.txt

# Copy project files
COPY . .

# Download models at build time
RUN python scripts/setup_metadata_system.py --download-only

# Expose ports
EXPOSE 8888 5000

# Start script
CMD ["/workspace/runpod/start_server.sh"]
