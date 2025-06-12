FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /workspace

# Copy and install Python dependencies first to leverage caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the rest of the project
COPY . /workspace

# Default command runs the training script using GPU if available
CMD ["python", "train_resnet.py"]
