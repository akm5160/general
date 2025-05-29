from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain_community.chat_models import AzureChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

class MeetingNotes(BaseModel):
    summary: str
    action_items: list[str]

@tool(args_schema=MeetingNotes)
def validate_meeting_notes(summary: str, action_items: list[str])-> MeetingNotes:
    """
    Validate and format meeting notes."""
    return {"summary":summary, "action_items":action_items}
llm = AzureChatOpenAI(
    deployment_name="gpt4o-mini",
    openai_api_version="2024-03-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)


agent=initialize_agent(
    tools=[validate_meeting_notes], 
    llm=llm, 
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)
transcript="""
Alice: Welcome everyone. today we need to finalize the Q3 roadmap.
Bob: I've emailed the updated feature list-please review by friday.
Carol: I'll set up the user-testing sessions next week.
Dan: Let's push the new UI mockups to staging on wednesday.
Alice: Great. Also, can someone compile the stakeholder feedback into a slide or deck?
Bob: I can handle the slide deck by Monday.
Alice: thanks, team. Meeting adjourned.
"""

response = agent.run(f"You are a meeting assistant."
                     f" 1. Summarize the meeting transcript below in exactly two sentences."
                     f" 2. Then list all action items mentioned, each as a separate bullet beginning with a dash."
                     f"\n\nReturn the result strictly as JSON with keys 'summary' and 'action_items'"
                     f"\n\n{transcript}")



####output
# > Entering new AgentExecutor chain...
# {
#   "summary": "The team met to finalize the Q3 roadmap and discussed various tasks related to it. Action items were assigned, including reviewing the feature list, setting up user-testing sessions, pushing UI mockups to staging, and compiling stakeholder feedback into a slide deck.",
#   "action_items": [
#     "- Review the updated feature list by Friday.",
#     "- Set up the user-testing sessions next week.",
#     "- Push the new UI mockups to staging on Wednesday.",
#     "- Compile the stakeholder feedback into a slide deck by Monday."
#   ]
# }
