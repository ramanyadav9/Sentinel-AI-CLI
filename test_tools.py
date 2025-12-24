import ollama
import json

host = "http://164.52.194.98:11434"
models = ["ministral-3:8b", "gemma3:latest"]

client = ollama.Client(host=host)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]

def test_model(model_name):
    print(f"\n--- Testing {model_name} ---")
    try:
        print("1. Basic Chat (No Tools)...")
        resp = client.chat(model=model_name, messages=[{"role": "user", "content": "Hi"}])
        print(f"   Success: {resp['message']['content'][:50]}...")
        
        print("2. Chat with Tools...")
        resp = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": "What is the weather in London?"}],
            tools=tools
        )
        msg = resp['message']
        if msg.get('tool_calls'):
            print("   SUCCESS: Model returned tool calls.")
            # Use safe printing for tool calls
            for tc in msg['tool_calls']:
                print(f"   Tool: {tc['function']['name']}({tc['function']['arguments']})")
        else:
            print("   FAILURE: No tool calls returned.")
            print(f"   Content: {msg.get('content')}")
            
    except Exception as e:
        print(f"   ERROR: {str(e)}")

for m in models:
    test_model(m)
