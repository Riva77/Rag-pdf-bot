import os
from dotenv import load_dotenv

# Load .env file if it exists
print(" Loading .env file...")
result = load_dotenv()
print(f" load_dotenv() result: {result}")

# Check if API key is set
api_key = os.getenv('GOOGLE_AI_API_KEY')
print(f"API Key found: {api_key is not None}")
if api_key:
    print(f"API Key length: {len(api_key)}")
    print(f"API Key starts with: {api_key[:10]}...")
    print(f"API Key ends with: ...{api_key[-10:]}")
else:
    print(" GOOGLE_AI_API_KEY not found!")
    print("Please set your API key in one of these ways:")
    print("1. Create a .env file with: GOOGLE_AI_API_KEY=your_key_here")
    print("2. Set environment variable: $env:GOOGLE_AI_API_KEY='your_key_here'")
