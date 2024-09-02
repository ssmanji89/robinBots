# Installation Guide

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

## Steps

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/robinBots.git
   cd robinBots
   ```

2. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy the `.env.example` file to `.env`
   - Fill in your Robinhood credentials and API keys

5. Initialize the database:
   ```
   python init_db.py
   ```

6. Run the initial setup script:
   ```
   ./setup.sh
   ```

Now you're ready to configure and run robinBots!
