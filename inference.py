import os
import json
import time
from openai import OpenAI
from env import CSVCleanerEnv, CSVAction

# ==========================================
# 1. SETUP LLM CONNECTION (SECURE)
# ==========================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# This looks for the Secret you added in Hugging Face Settings
API_KEY = os.getenv("OPENAI_API_KEY") 

def run_baseline(task_level="easy"):
    if not API_KEY:
        print("ERROR: API_KEY is missing. Add it to Settings > Secrets in Hugging Face!")
        return

    # Initialize the LLM client
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    
    # Initialize your custom environment
    env = CSVCleanerEnv(task_name=task_level)
    obs = env.reset()
    
    benchmark = "openenv-csv-cleaner"
    
    print(f"[START] task=csv-cleaning-{task_level} env={benchmark} model={MODEL_NAME}")
    
    done = False
    step_count = 0
    rewards_history = []
    
    system_prompt = """You are an AI data cleaning agent. 
    Analyze the 'null_counts', 'columns', and 'head' to determine what is wrong.
    You MUST respond with a valid JSON object:
    {
        "operation": "drop_na" | "format_date" | "fix_typo" | "submit",
        "column": "string (optional)",
        "target_value": "string (optional)",
        "new_value": "string (optional)"
    }"""

    while not done and step_count < 10:
        step_count += 1
        error_msg = "null"
        action_str = "none"
        
        try:
            user_prompt = f"Current State: {obs.model_dump_json()}"
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            
            raw_reply = response.choices[0].message.content
            clean_reply = raw_reply.replace("```json", "").replace("```", "").strip()
            action_data = json.loads(clean_reply)
            
            action = CSVAction(**action_data)
            action_str = f"{action.operation}({action.column or ''})"
            
            obs, reward, done, info = env.step(action)
            
        except Exception as e:
            reward = -0.5
            done = True
            error_msg = f"'{str(e)}'"
            action_str = "ai_error"
            
        rewards_history.append(f"{reward:.2f}")
        
        done_str = "true" if done else "false"
        print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={done_str} error={error_msg}")
        
    success_str = "true" if sum(float(r) for r in rewards_history) > 0 else "false"
    rewards_joined = ",".join(rewards_history)
    print(f"[END] success={success_str} steps={step_count} rewards={rewards_joined}")

if __name__ == "__main__":
    run_baseline("easy")
    
    # Keep the script alive so Hugging Face doesn't show "Runtime Error"
    print("Inference finished. Keeping container alive for logs...")
    while True:
        time.sleep(60)
