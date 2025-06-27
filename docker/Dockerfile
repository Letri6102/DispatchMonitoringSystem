# Use Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app into the container
COPY . /app/

# Expose the app's port (if using Streamlit or Flask, change accordingly)
EXPOSE 8501

# Command to run the app (adjust based on your entry point)
CMD ["python", "modules/app.py"]
