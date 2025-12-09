import httpx
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OLLAMA_API_KEY")
host = os.getenv("OLLAMA_HOST", "https://ollama.com")
headers = {"Authorization": f"Bearer {api_key}"}

print(f"Testing connection to: {host}")
print(f"Using API Key: {api_key[:5]}...")

try:
    print("\nAttempting direct httpx request...")
    resp = httpx.get(f"{host}/api/tags", headers=headers, timeout=10.0)
    print(f"Status Code: {resp.status_code}")
    print("Response Headers:", resp.headers)
    print("Success! Python can reach Ollama Cloud.")
except Exception as e:
    print(f"\nFAILED to connect: {e}")
    import traceback
    traceback.print_exc()

print("\n-------------------")
print("Environment Info:")
print(f"HTTP_PROXY: {os.environ.get('HTTP_PROXY')}")
print(f"HTTPS_PROXY: {os.environ.get('HTTPS_PROXY')}")
