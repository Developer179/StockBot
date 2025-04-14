# Stock Market API

A Flask RESTful API for accessing stock market data and AI-powered analysis.

## Project Structure

This project follows a modular structure to separate concerns and make the code more maintainable:

```
stock_api/
├── app/                  # Application package
│   ├── __init__.py       # Flask app initialization
│   ├── routes/           # Route handlers
│   ├── models/           # Data models
│   ├── utils/            # Utility functions
│   └── config.py         # Configuration settings
├── run.py                # Application entry point
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Configure database connection in `app/utils/db.py`

## Running the Application

```
python run.py
```

The API will be available at http://localhost:5000

## API Endpoints

- `/search?q={query}` - Search for companies
- `/start-session` - Start a company data session
- `/ask` - Ask a question about a company
- `/start-screener-session` - Start a screener session
- `/screener-question` - Ask a question about screeners
- `/clear-session` - Clear a session
- `/debug-db` - Debug database connection

## CORS Support

This API has CORS enabled for http://localhost:3000 to support frontend development.

## Frontend Integration

You can connect to this API from a React application running on http://localhost:3000.

## Troubleshooting

- If you encounter CORS issues, check that the frontend is running on the expected URL
- Database connection errors can be diagnosed using the `/debug-db` endpoint
- For issues with the AI model, ensure Ollama is running with the Gemma model available