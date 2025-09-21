from google.adk.agents import Agent

root_agent = Agent(
    name="greeting_agent",
    model = "gemini-2.0-flash",
    description = "Greeting agent",
    instrcution = """
    You are a helpful assistant that greets the user.
    Ask you the user's name and greets them by name.
    """
)
