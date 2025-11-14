import yaml
import os
from dotenv import load_dotenv
from haystack.dataclasses import ChatMessage
from client.utils.llm_factory import create_llm

# --- CONFIGURATION ---
PROVIDER_TO_TEST = "gemini"  # Prueba "gemini" o "huggingface"
# ---------------------

def run_test():
    print("--- Starting LLM Factory Test ---")
    load_dotenv()
    print("✅ Loaded .env file.")

    try:
        with open("config/models.yaml", 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Loaded config/models.yaml.")
    except Exception as e:
        print(f"❌ Error loading YAML: {e}")
        return

    try:
        provider_config = config['providers'][PROVIDER_TO_TEST]
        model_name = provider_config['model']
        print(f"🔬 Testing provider: '{PROVIDER_TO_TEST}' with model: '{model_name}'")
    except KeyError:
        print(f"❌ Error: Provider '{PROVIDER_TO_TEST}' not found in models.yaml.")
        return

    # 4. --- CALL YOUR FACTORY ---
    try:
        llm = create_llm(
            provider=PROVIDER_TO_TEST,
            model=model_name
        )
        print(f"✅ Factory created LLM component: {type(llm)}")
    except Exception as e:
        print(f"❌ Error creating LLM from factory: {e}")
        return

    # 5. --- RUN THE LLM COMPONENT DIRECTLY ---
    try:
        print("\nAttempting to call LLM API directly...")
        
        messages = [ChatMessage.from_user("Hello, in one sentence, who are you?")]
        
        # Call the component directly using its run() method
        result = llm.run(messages=messages)
        
        # Extract the response from the result
        # The result structure is typically: {"replies": [ChatMessage, ...]}
        if "replies" in result and len(result["replies"]) > 0:
            assistant_message = result["replies"][0]
            assistant_response = assistant_message.content if hasattr(assistant_message, 'content') else str(assistant_message)
        else:
            # Fallback: check if messages are in the result directly
            assistant_response = str(result)
        
        print("\n--- TEST SUCCEEDED ---")
        print(f"✅ LLM Response: {assistant_response}")
        print("---------------------------------")
        
    except Exception as e:
        print(f"\n❌ Error running the LLM component: {e}")
        import traceback
        traceback.print_exc()
        print("--- TEST FAILED ---")

if __name__ == "__main__":
    run_test()