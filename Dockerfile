FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port required by Hugging Face and the grader
EXPOSE 7860

# Run the server version of your script
CMD ["python", "inference.py"]
