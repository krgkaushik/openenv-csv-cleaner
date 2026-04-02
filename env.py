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
    target_value: Optional[str] = Field(default=None, description="The bad value to replace (for fix_typo)")
    new_value: Optional[str] = Field(default=None, description="The correct value (for fix_typo)")

class CSVObservation(BaseModel):
    """What the AI agent can see about the data."""
    head: str = Field(description="A string representation of the first 5 rows of the dataset")
    columns: List[str] = Field(description="List of all column names")
    null_counts: Dict[str, int] = Field(description="How many missing values exist in each column")
    feedback: str = Field(description="System feedback from the last action")

# ==========================================
# 2. CREATE THE ENVIRONMENT CLASS
# ==========================================

class CSVCleanerEnv:
    def __init__(self, task_name="easy"):
        self.task_name = task_name.lower()
        self.current_step = 0
        self.max_steps = 10
        
        # A master dataset with missing values, messy dates, and a typo ("Saels")
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
        reward = 0.0
        done = False
        feedback = ""

        try:
            # -- ACTION REPERTOIRE --
            if action.operation == "drop_na":
                self.df = self.df.dropna(subset=[action.column] if action.column else None)
                feedback = "Dropped missing values."
                
            elif action.operation == "format_date" and action.column:
                self.df[action.column] = pd.to_datetime(self.df[action.column], errors='coerce').dt.strftime('%Y-%m-%d')
                feedback = "Formatted dates to YYYY-MM-DD."
                
            elif action.operation == "fix_typo" and action.column and action.target_value:
                self.df[action.column] = self.df[action.column].replace(action.target_value, action.new_value)
                feedback = f"Replaced '{action.target_value}' with '{action.new_value}'."

            elif action.operation == "submit":
                done = True
                
                # -- THE GRADER LOGIC (0.0 to 1.0) --
                if self.task_name == "easy":
                    # Easy: Just drop nulls
                    score = 1.0 if self.df.isnull().sum().sum() == 0 else 0.0
                    feedback = "Easy Grader: Dataset has no nulls." if score == 1.0 else "Easy Grader Failed: Nulls still exist."
                
                elif self.task_name == "medium":
                    # Medium: Dates must be perfectly formatted strings (length 10: YYYY-MM-DD)
                    dates_clean = all(isinstance(str(x), str) and len(str(x)) == 10 for x in self.df["Date"].dropna())
                    score = 1.0 if dates_clean else 0.0
                    feedback = "Medium Grader: Dates formatted perfectly." if score == 1.0 else "Medium Grader Failed: Dates messy."

                elif self.task_name == "hard":
                    # Hard: Typo in department "Saels" must be gone, but row cannot be deleted
                    typo_fixed = "Saels" not in self.df["Department"].values
                    rows_kept = len(self.df) == len(self.original_df)
                    score = 1.0 if (typo_fixed and rows_kept) else 0.0
                    feedback = "Hard Grader: Typo fixed without dropping rows!" if score == 1.0 else "Hard Grader Failed."
                
                else:
                    score = 0.0
                    feedback = "Unknown task."
                
                reward = score

            else:
                feedback = "Invalid action."
                reward = -0.1

        except Exception as e:
            feedback = f"Error: {str(e)}"
            reward = -0.5

        if self.current_step >= self.max_steps:
            done = True
            feedback += " Max steps reached."

        obs = self.state()
        obs.feedback = feedback
        info = {"step": self.current_step}

        return obs, reward, done, info