# Use Python 3.13
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Command to run your inference script
CMD ["python", "inference.py"]