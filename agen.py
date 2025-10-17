from typing import Dict, Any
from langgraph.graph import StateGraph, END
from google import genai
from dotenv import load_dotenv
import os

load_dotenv(override=True)
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
# ------------------------------
# 1. Gemini LLM Setup
# ------------------------------
#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")  # Make sure GOOGLE_API_KEY is set

# ------------------------------
# 2. Memory
# ------------------------------
class AgentMemory(dict):
    """Persistent memory for requirements, drafts, and updates."""
    def remember(self, key: str, value: Any):
        self[key] = value

    def recall(self, key: str, default=None):
        return self.get(key, default)

memory = AgentMemory()

# ------------------------------
# 3. Helper Functions
# ------------------------------
def gemini_call(prompt: str) -> str:
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    if hasattr(response, "candidates") and response.candidates:
        return response.candidates[0].content.parts[0].text
    return str(response)


def generate_draft_with_gemini(parsed: Dict[str, Any]) -> str:
    prompt = f"""
    User requirements parsed: {parsed}

    Create a draft workflow and list of tools needed to build the agent.
    Do not return JSON yet, just human-readable details.
    """
    return gemini_call(prompt)

def update_workflow_with_gemini(draft: str, change_request: str) -> str:
    prompt = f"""
    Current draft workflow:
    {draft}

    User requested changes:
    {change_request}

    Update only the necessary parts of the workflow, don't rebuild everything.
    """
    return gemini_call(prompt)

def finalize_json_with_gemini(draft: str) -> str:
    prompt = f"""
    Convert the following agent workflow and tools into a clean JSON format:

    {draft}
    """
    return gemini_call(prompt)

# ------------------------------
# 4. Node Functions
# ------------------------------
def parse_requirements(state: Dict[str, Any]):
    user_input = state["requirements"]
    parsed = {"goal": user_input}  # Simplified parsing; can use Gemini for more detail
    memory.remember("requirements", user_input)
    memory.remember("parsed", parsed)
    return {"parsed": parsed}

def draft_workflow(state: Dict[str, Any]):
    parsed = memory.recall("parsed")
    workflow = generate_draft_with_gemini(parsed)
    memory.remember("draft", workflow)
    return {"draft": workflow}

def present_workflow(state: Dict[str, Any]):
    draft = memory.recall("draft")
    print("\n📋 Proposed Workflow & Tools:")
    print(draft)
    user_feedback = input("\nDo you confirm this? (yes / specify changes): ")
    return {"user_feedback": user_feedback}

def confirm_node(state: Dict[str, Any]):
    feedback = state.get("user_feedback", "")
    if feedback.lower().startswith("yes"):
        memory.remember("confirmed", True)
        return {"confirmed": True}
    else:
        memory.remember("confirmed", False)
        memory.remember("user_feedback", feedback)
        return {"confirmed": False, "user_feedback": feedback}

def update_workflow(state: Dict[str, Any]):
    draft = memory.recall("draft")
    change_request = memory.recall("user_feedback")
    updated = update_workflow_with_gemini(draft, change_request)
    memory.remember("draft", updated)
    return {"draft": updated}

def finalize_json(state: Dict[str, Any]):
    draft = memory.recall("draft")
    final_json = finalize_json_with_gemini(draft)
    memory.remember("final_json", final_json)
    return {"final_json": final_json}

# ------------------------------
# 5. Build Graph
# ------------------------------
graph = StateGraph(dict)

graph.add_node("parse", parse_requirements)
graph.add_node("draft", draft_workflow)
graph.add_node("present", present_workflow)
graph.add_node("confirm", confirm_node)
graph.add_node("update", update_workflow)
graph.add_node("finalize", finalize_json)

graph.set_entry_point("parse")
graph.add_edge("parse", "draft")
graph.add_edge("draft", "present")
graph.add_edge("update", "present")

def confirm_router(state: Dict[str, Any]):
    return "finalize" if state.get("confirmed") else "update"

graph.add_conditional_edges("confirm", confirm_router)
graph.add_edge("present", "confirm")
graph.add_edge("finalize", END)

agent_graph = graph.compile()

# ------------------------------
# 6. Chat-Style Run Loop
# ------------------------------
def run_chat_agent():
    print("🤖 Welcome to the Agent Builder Chat! Type 'exit' anytime to quit.\n")
    
    # Step 1: Get initial requirements
    user_req = input("📝 Enter your agent requirements: ")
    if user_req.lower() == "exit":
        return
    state = {"requirements": user_req}
    
    # Step 2: Keep running until final JSON is generated
    while True:
        state = agent_graph.invoke(state)
        final_json = memory.recall("final_json")
        if final_json:
            print("\n✅ Final Workflow JSON:")
            print(final_json)
            break

        # If user says exit during feedback
        user_feedback = state.get("user_feedback", "")
        if user_feedback.lower() == "exit":
            print("\nExiting Agent Builder. Goodbye!")
            break

if __name__ == "__main__":
    run_chat_agent()
    genai_client.close()