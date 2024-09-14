# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Streamlit uses 8501 by default)
EXPOSE 8501

# Set environment variables for OpenAI and Search API keys
ENV OPENAI_API_KEY="sk-BlVlRBU8ErgZFGemKSJW2DZfYb3qkN2gY38zhUvzVRT3BlbkFJTU0x9gJEWUEFAbOaOfsJwtuF2ZxQ7gIc-OX5ZbzJkA"
ENV SEARCHAPI_API_KEY="JQc1NRAHV1jdYNoCQQ11bjhV"

# Command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.enableCORS", "false"]
