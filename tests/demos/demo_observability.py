from dotenv import load_dotenv

load_dotenv()

import openai

from agency_swarm import Agency, Agent
from agency_swarm.util.oai import get_tracker, set_tracker

TRACKER = "local"
# TRACKER = "langfuse"

if TRACKER == "local":
    # Test SQLite configuration
    set_tracker("local")
elif TRACKER == "langfuse":
    # Test Langfuse configuration
    set_tracker("langfuse")
    openai.langfuse_auth_check()

# Create multiple agents with different roles
ceo = Agent(
    name="CEO",
    description="Manages projects and coordinates between team members",
    temperature=0.5,
)

developer = Agent(
    name="Developer",
    description="Implements technical solutions and writes code",
    temperature=0.3,
)

analyst = Agent(
    name="Data Analyst",
    description="Analyzes data and provides insights",
    temperature=0.4,
)

# Create agency with communication flows
agency = Agency(
    [
        ceo,  # CEO is the entry point
        [ceo, developer],  # CEO can communicate with Developer
        [ceo, analyst],  # CEO can communicate with Analyst
        [developer, analyst],  # Developer can communicate with Analyst
    ]
)

# Run the demo with Gradio interface
agency.demo_gradio()
# Run the CLI demo
# agency.run_demo()

print(get_tracker().get_total_tokens())
