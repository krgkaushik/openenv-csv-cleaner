FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into the container
COPY . .

# Expose the specific port Hugging Face Spaces uses
EXPOSE 7860
ENV PORT=7860

# Run the Python script directly, which starts Uvicorn and the background thread
CMD ["python", "-m", "server.app"]
