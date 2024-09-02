# Use the official Python 3.11 image as a base
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the .env file into the container
COPY .env .env

# Install required Python packages from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Modify the pandas_ta package as specified
RUN sed -i 's/from numpy import NaN as npNaN/from numpy import nan as npNaN/' /usr/local/lib/python3.11/site-packages/pandas_ta/momentum/squeeze_pro.py

# Copy the rest of the application code
COPY . .

# Load environment variables from .env file
RUN export $(cat .env | xargs)

# Specify the command to run the application
CMD ["python", "src/main.py"]
