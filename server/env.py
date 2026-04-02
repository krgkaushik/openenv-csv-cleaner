import pandas as pd
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple, Any

# ==========================================
# 1. DEFINE THE PYDANTIC MODELS
# ==========================================
class CSVAction(BaseModel):
    """What the AI agent is allowed to do."""
    operation: str = Field(description="Valid operations: 'drop_na', 'format_date', 'fix_typo', or 'submit'")
    column: Optional[str] = Field(default=None, description="The specific column to clean")
    target_value: Optional[str] = Field(default=None, description="The bad value to replace")
    new_value: Optional[str] = Field(default=None, description="The correct value")

class CSVObservation(BaseModel):
    """What the AI agent can see about the data."""
    head: str = Field(description="First 5 rows representation")
    columns: List[str] = Field(description="List of all column names")
    null_counts: Dict[str, int] = Field(description="Null counts per column")
    feedback: str = Field(description="System feedback from last action")

# ==========================================
# 2. CREATE THE ENVIRONMENT CLASS
# ==========================================
class CSVCleanerEnv:
    def __init__(self, task_name="easy"):
        self.task_name = task_name.lower()
        self.current_step = 0
        self.max_steps = 10
        
        # Initial dataset for cleaning
        self.original_df = pd.DataFrame({
            "Name": ["Alice", "Bob", None, "Dave", "Eve"],
            "Department": ["Sales", "HR", "Sales", "Saels", "IT"], 
            "Date": ["12/01/2023", "2023-12-02", "12/03/2023", "12-04-2023", "2023/12/05"]
        })
        self.df = self.original_df.copy()

    def state(self) -> CSVObservation:
        return CSVObservation(
            head=self.df.head().to_string(),
            columns=self.df.columns.tolist(),
            null_counts=self.df.isnull().sum().to_dict(),
            feedback="Current state loaded."
        )

    def reset(self) -> CSVObservation:
        self.df = self.original_df.copy()
        self.current_step = 0
        return self.state()

    def step(self, action: CSVAction) -> Tuple[CSVObservation, float, bool, Dict[str, Any]]:
        self.current_step += 1
        reward = 0.0  # Initialized as float
        done = False
        feedback = ""

        try:
            # -- ACTION REPERTOIRE --
            if action.operation == "drop_na":
                self.df = self.df.dropna(subset=[action.column] if action.column else None)
                feedback = "Dropped missing values."
            
            elif action.operation == "format_date" and action.column:
                self.df[action.column] = pd.to_datetime(self.df[action.column], errors='coerce').dt.strftime('%Y-%m-%d')
                feedback = "Formatted dates."
            
            elif action.operation == "fix_typo" and action.column:
                self.df[action.column] = self.df[action.column].replace(action.target_value, action.new_value)
                feedback = f"Fixed typo in {action.column}."

            elif action.operation == "submit":
                done = True
                # -- THE GRADER LOGIC (Strict Floats) --
                if self.task_name == "easy":
                    reward = 1.0 if self.df.isnull().sum().sum() == 0 else 0.0
                elif self.task_name == "medium":
                    dates_clean = all(len(str(x)) == 10 for x in self.df["Date"].dropna())
                    reward = 1.0 if dates_clean else 0.0
                elif self.task_name == "hard":
                    typo_fixed = "Saels" not in self.df["Department"].values
                    rows_kept = len(self.df) == len(self.original_df)
                    reward = 1.0 if (typo_fixed and rows_kept) else 0.0
                else:
                    reward = 0.0
                feedback = f"Submitted. Score: {reward}"
            else:
                feedback = "Invalid operation."
                reward = -0.1

        except Exception as e:
            feedback = f"Error: {str(e)}"
            reward = -0.5

        if self.current_step >= self.max_steps:
            done = True
        
        obs = self.state()
        obs.feedback = feedback
        
        # Ensure 'info' is a dict and types are strict for the grader
        return obs, float(reward), bool(done), {"step": int(self.current_step)}
