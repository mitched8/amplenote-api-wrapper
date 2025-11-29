import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
API_URL = "http://localhost:8000/api/notes"

# 1. Get API Key from environment
api_key = os.getenv("OPENAI_API_KEY")

# Debug: Print partial key to verify loading
if api_key:
    print(f"DEBUG: Loaded API key starting with: {api_key[:8]}...")
else:
    print("DEBUG: No API key found in environment.")

if not api_key or api_key == "sk-your-api-key-here":
    print("❌ Error: OPENAI_API_KEY not found in .env or environment variables.")
    print("Please edit the .env file with your actual OpenAI API key.")
    exit(1)

client = OpenAI(api_key=api_key)

def get_amplenote_data(days=7):
    print(f"📡 Fetching notes from {API_URL} (last {days} days)...")
    try:
        response = requests.get(f"{API_URL}?days={days}")
        
        # Check for authentication error (HTTP 401)
        if response.status_code == 401:
            error_data = response.json()
            print("\n⚠️  Authentication Required!")
            print(f"Message: {error_data.get('message')}")
            if "auth_url" in error_data:
                print(f"URL: {error_data['auth_url']}")
            return None
            
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching notes: {e}")
        return None

def summarize_with_openai(notes_data):
    if not notes_data or "notes" not in notes_data:
        print("No notes found to summarize.")
        return

    notes_list = notes_data["notes"]
    count = notes_data.get("count", len(notes_list))
    
    print(f"📄 Found {count} notes. Preparing prompt for OpenAI...")

    # format notes for the prompt
    notes_text = ""
    for note in notes_list:
        name = note.get('name', 'Untitled')
        uuid = note.get('uuid')
        tags = ", ".join(note.get('tags', []))
        notes_text += f"- {name} (Tags: {tags})\n"

    prompt = f"""
    Here is a list of my recent notes from AmpleNote:
    
    {notes_text}
    
    Please analyze these note titles and tags. 
    1. What appear to be the main topics I'm working on?
    2. Are there any specific projects or daily habits visible?
    3. Provide a 2-sentence summary of my recent activity.
    """

    # Using gpt-4o as the latest flagship model. 
    # Note: "5.1" is likely a typo or future model; sticking to gpt-4o for now.
    model_name = "gpt-5.1"
    print(f"🤖 Sending request to OpenAI ({model_name})...")
    
    try:
        completion = client.chat.completions.create(
            model=model_name, 
            messages=[
                {"role": "system", "content": "You are a helpful productivity assistant analyzing note metadata."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return None

if __name__ == "__main__":
    # 1. Get Data from your Local API Wrapper
    data = get_amplenote_data(days=7)
    
    # 2. Send to OpenAI
    if data:
        summary = summarize_with_openai(data)
        if summary:
            print("\n" + "="*40)
            print("   OPENAI SUMMARY OF YOUR AMPLENOTE   ")
            print("="*40 + "\n")
            print(summary)
            print("\n" + "="*40)
