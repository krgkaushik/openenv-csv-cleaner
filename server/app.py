import os
import json
import time
import uvicorn
import threading
from fastapi import FastAPI
from openai import OpenAI

# Updated import to work from the root directory
from server.env import CSVCleanerEnv, CSVAction

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
app = FastAPI()
env = CSVCleanerEnv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
API_KEY = os.getenv("OPENAI_API_KEY")

# ==========================================
# 2. FASTAPI ENDPOINTS (FOR THE GRADER)
# ==========================================
@app.get("/")
async def health_check():
    return {"status": "running", "message": "OpenEnv CSV Cleaner is active"}

@app.post("/reset")
async def reset():
    """Handles the Reset POST check from the grader."""
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
async def step(action: CSVAction):
    """Handles the Step POST actions with strict typing for validation."""
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(reward),
        "done": bool(done),
        "info": info if info else {}
    }

# ==========================================
# 3. AI AGENT LOGIC (THE BASELINE)
# ==========================================
def run_baseline(task_level="easy"):
    time.sleep(5) # Delay to allow server to initialize
    
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY is missing in HF Secrets!")
        return

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    agent_env = CSVCleanerEnv(task_name=task_level)
    obs = agent_env.reset()
    
    print(f"[START] task=csv-cleaning-{task_level} env=openenv-csv-cleaner model={MODEL_NAME}")
    
    done, step_count, rewards_history = False, 0, []
    system_prompt = "You are an AI data cleaning agent. Respond ONLY with raw JSON matching the CSVAction schema."

    while not done and step_count < 10:
        step_count += 1
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Current State: {obs.model_dump_json()}"}
                ],
                temperature=0
            )
            raw_reply = response.choices[0].message.content
            clean_reply = raw_reply.replace("```json", "").replace("```", "").strip()
            action_data = json.loads(clean_reply)
            action = CSVAction(**action_data)
            
            obs, reward, done, _ = agent_env.step(action)
            action_str = f"{action.operation}({action.column or ''})"
            error_msg = "null"
        except Exception as e:
            reward, done, error_msg, action_str = -0.5, True, f"'{str(e)}'", "ai_error"
            
        rewards_history.append(f"{reward:.2f}")
        print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={'true' if done else 'false'} error={error_msg}")
        
    print(f"[END] success={'true' if sum(float(r) for r in rewards_history) > 0 else 'false'} steps={step_count} rewards={','.join(rewards_history)}")

# ==========================================
# 4. ENTRY POINT
# ==========================================
def main():
    agent_thread = threading.Thread(target=run_baseline, args=("easy",))
    agent_thread.daemon = True
    agent_thread.start()

    port = int(os.environ.get("PORT", 7860))
    print(f"Starting server on port {port}...")
    # Running uvicorn directly from the script
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
