# Use the official Python image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install the required libraries
RUN pip install -r requirements.txt

# Expose the Flask port
EXPOSE 5001

# Run the Flask application
CMD ["python", "app.py"]