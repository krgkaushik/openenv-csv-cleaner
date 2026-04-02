import os
import json
from openai import OpenAI
from env import CSVCleanerEnv, CSVAction

# ==========================================
# 1. SETUP LLM CONNECTION
# ==========================================
# The judges will automatically inject these variables during the test
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "dummy-key-for-testing")

def run_baseline(task_level="easy"):
    # Initialize the LLM client
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    
    # Initialize your custom environment with the specific task
    env = CSVCleanerEnv(task_name=task_level)
    obs = env.reset()
    
    benchmark = "openenv-csv-cleaner"
    
    # [START] Required by automated grader
    print(f"[START] task=csv-cleaning-{task_level} env={benchmark} model={MODEL_NAME}")
    
    done = False
    step_count = 0
    rewards_history = []
    
    # Give the AI strict instructions on how to behave
    system_prompt = """You are an AI data cleaning agent. 
    You will be given a JSON representation of a messy Pandas DataFrame.
    Analyze the 'null_counts', 'columns', and 'head' to determine what is wrong.
    You MUST respond with a valid JSON object matching this exact schema:
    {
        "operation": "drop_na" | "format_date" | "fix_typo" | "submit",
        "column": "string (optional)",
        "target_value": "string (optional)",
        "new_value": "string (optional)"
    }
    Only output the raw JSON object. Do not include markdown formatting or explanations."""

    while not done and step_count < 10:
        step_count += 1
        error_msg = "null"
        action_str = "none"
        
        try:
            # 1. Turn the environment observation into a string for the AI
            user_prompt = f"Current State: {obs.model_dump_json()}"
            
            # 2. Ask the LLM what action to take
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1 # Keep it low so the AI is highly deterministic
            )
            
            # 3. Clean and parse the AI's response
            raw_reply = response.choices[0].message.content
            clean_reply = raw_reply.replace("```json", "").replace("```", "").strip()
            action_data = json.loads(clean_reply)
            
            # 4. Convert it to our strict Pydantic model
            action = CSVAction(**action_data)
            action_str = f"{action.operation}({action.column or ''})"
            
            # 5. Take the step in the environment
            obs, reward, done, info = env.step(action)
            
        except Exception as e:
            # If the AI hallucinates bad JSON or crashes, we penalize it
            reward = -0.5
            done = True
            error_msg = f"'{str(e)}'"
            action_str = "ai_error"
            
        rewards_history.append(f"{reward:.2f}")
        
        # [STEP] Required by automated grader
        done_str = "true" if done else "false"
        print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={done_str} error={error_msg}")
        
    # [END] Required by automated grader
    success_str = "true" if sum(float(r) for r in rewards_history) > 0 else "false"
    rewards_joined = ",".join(rewards_history)
    print(f"[END] success={success_str} steps={step_count} rewards={rewards_joined}")

if __name__ == "__main__":
    # You can change this to "medium" or "hard" to test the other graders!
    run_baseline("easy")