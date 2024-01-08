# Use the specified CUDA image as the base image
FROM gpuci/cuda:11.5.0-devel-ubuntu20.04

# Set the timezone (optional)
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Add NVIDIA GPG key
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# Update and install some basic tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install FastAPI and Uvicorn directly
RUN pip3 install --upgrade pip && \
    pip3 install fastapi uvicorn

# Copy requirements file into the container
COPY requirements.txt /tmp/

# Install other Python dependencies from requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Copy the rest of the application into the container
COPY . /app

# Set up the working directory
WORKDIR /app

# Command to run on container start: run FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
