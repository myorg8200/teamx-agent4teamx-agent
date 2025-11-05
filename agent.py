from bedrock_agentcore import BedrockAgentCoreApp
from crewai import Agent, Task, Crew, Process
import os

app = BedrockAgentCoreApp()

# Set AWS region for litellm (used by CrewAI)
os.environ["AWS_DEFAULT_REGION"] = os.environ.get("AWS_REGION", "us-west-2")

# Create research agent - will be customized per team
researcher = Agent(
    role="Research Assistant",
    goal="Provide helpful and accurate information",
    backstory="You are a knowledgeable research assistant with expertise in many domains",
    verbose=False,
    llm="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # Template placeholder - will be replaced
    max_iter=2
)

@app.entrypoint
def invoke(payload):
    user_message = payload.get("prompt", "Hello!")
    
    # Create a task for the agent
    task = Task(
        description=user_message,
        agent=researcher,
        expected_output="A helpful and informative response"
    )
    
    # Create and run the crew
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False
    )
    
    result = crew.kickoff()
    return {"result": result.raw}

if __name__ == "__main__":
    app.run()
