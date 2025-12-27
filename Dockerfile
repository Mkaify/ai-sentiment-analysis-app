# 1. Use an official lightweight Python image
FROM python:3.10-slim

# 2. Set environment variables to ensure clean output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system dependencies for torch and building tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy and install Python requirements
# We use the CPU-only version of torch to keep the image size small
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy all project files into the container
COPY . .

# 7. Expose the port FastAPI runs on
EXPOSE 8000

# 8. Command to run the application
# We use 0.0.0.0 so the container is accessible from outside
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]