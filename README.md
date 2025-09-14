# Fake-News-Killer-API
The backend powerhouse for the Fake News Killer project. This repository contains the Python and Flask server that hosts our AI model, exposing a REST API to analyze text for signs of misinformation. It serves as the core intelligence for the Fake News Killer browser extension.

Fake News Killer - AI Model & API
This repository contains the backend powerhouse for the Fake News Killer project. It's a Python and Flask web service that hosts our custom-trained AI model and exposes a REST API to analyze text for patterns of misinformation. This server provides the core intelligence for the Fake News Killer browser extension.

How It Works
The application is a simple yet powerful web server designed to do one thing: receive a piece of text and return a credibility prediction.
Request Received: The server listens for POST requests on the /predict endpoint.
Data Processing: The incoming JSON data containing the text is parsed and sanitized.
AI Model Inference: The text is passed to our pre-trained machine learning model, which analyzes its linguistic patterns.
Prediction Generated: The model outputs a prediction, classifying the text as either "REAL" or "FAKE".
Response Sent: The server packages this prediction into a JSON object and sends it back to the client (the browser extension).

API Endpoint
The server exposes a single, primary endpoint for all prediction tasks.
`POST /predict`
This endpoint analyzes a given block of text and returns a prediction.

Request Body:
`{
  "text": "Your news article text goes here..."
}`
Success Response (200 OK):

`{
  "prediction": "REAL"
}`
or
`{
  "prediction": "FAKE"
}`
Error Response (400 Bad Request):

`{
  "error": "No text provided for analysis."
}`

# Technology Stack
Backend Framework: Flask - A lightweight and flexible Python web framework.
WSGI Server: Gunicorn - A production-ready web server for running Python applications.
Core AI Libraries: scikit-learn, pandas, and numpy.

# Getting Started
Follow these instructions to get a copy of the project up and running on your local machine for development and testing.

Prerequisites
Python 3.8 or higher
pip (Python package installer)

Local Setup
Clone the repository:
Bash
`git clone https://github.com/your-username/fake-news-killer-api.git`
`cd fake-news-killer-api`
Create and activate a virtual environment:
Bash
For macOS/Linux
`python3 -m venv venv`
`source venv/bin/activate`

For Windows
`python -m venv venv`
`.\venv\Scripts\activate`

Install the required dependencies:
Bash
`pip install -r requirements.txt`

Run the Flask development server:
`python api_service.py`

Deployment
This application is designed to be deployed as a Web Service on cloud platforms like Render or Heroku.
Build Command: `pip install -r requirements.txt`
Start Command: `gunicorn app:app`
