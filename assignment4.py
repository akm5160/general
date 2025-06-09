import os
import json
import openai
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from openai import AzureOpenAI
from langchain_core.tools import tool
import functools
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

# ========== Set Your Environment Variables ==========
# You can also set them manually like below for testing

embedding_endpoint="https://amit-openai-rag.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-05-15"
embdding_api_key="8pyXXXXX"
model_openai_endpoint="https://amit-openai-rag.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview"
model_api_key="8pyvXXXXX"

os.environ["AZURE_OPENAI_API_KEY"] = model_api_key
os.environ["AZURE_OPENAI_ENDPOINT"] = model_api_key
os.environ["PINECONE_API_KEY"] = "pcsk_XXXXX"
os.environ["PINECONE_REGION"] = "us-east-1"  # or your specific region
os.environ["PINECONE_INDEX_NAME"] = "kb-index"



# ========== Azure OpenAI Setup ======
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

# ========== Pinecone Setup ==========
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
region = os.getenv("PINECONE_REGION")

# Create index if not already exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=region)
    )

index = pc.Index(index_name)

# ========== Load JSON Data ==========
with open("self_critique_loop_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

import pandas as pd

embedding_client = AzureOpenAI(
    api_key=embdding_api_key,
    api_version="2024-02-15-preview",
    azure_endpoint=embedding_endpoint
)

# ========== Embed and Upsert ==========
batch_size = 10
for i in tqdm(range(0, len(data), batch_size), desc="Embedding and Upserting"):
    batch = data[i:i + batch_size]
    texts = [item["answer_snippet"] for item in batch]

    response = embedding_client.embeddings.create(
    model="text-embedding-ada-002",
    input=texts
)
    # print(response)
    embeddings = [rec.embedding for rec in response.data]

    vectors = [
        {
            "id": item["doc_id"],
            "values": emb,
            "metadata": {
                "source": item["source"],
                "last_updated": item["last_updated"]
            }
        }
        for item, emb in zip(batch, embeddings)
    ]
    index.upsert(vectors=vectors)


@tool
def retrieve_tool(query: Annotated[str, "tool is used retrive relevant information from the embedded data from the vectorstore."]):
    """
     tool is used retrive relevant information from the embedded data from the vectorstore.
    """
    response=embedding_client.embeddings.create(input=[query], model="text-embedding-ada-002")
    query_embed = response.data[0].embedding
    results = index.query(vector=query_embed, top_k=3, include_metadata=True)
    return results


def print_results(results,data):
    print("\nTop Matches:\n")
    for match in results["matches"]:
        print(f"Score: {match['score']:.4f}")
        print(f"Doc ID: {match['id']}")
        print(f"Source: {match['metadata'].get('source')}")
        print(f"answer_snippet: {data[data['doc_id']==match['id']]['answer_snippet'].values}")
        print(f"Last Updated: {match['metadata'].get('last_updated')}")
        print("-" * 40)








class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: Annotated[str, "the sender of last message"]
    query: Annotated[str, "User question"]
    kb_hits: Annotated[list, "Knowledge base search results"]
    last_answer: Annotated[str,"Last output sent by the last executed node"]
    generation_answer: Annotated[str,"Generation stage answer"]
    critique_result:Annotated[str, "Critique stage result: REFINE or COMPLETE"]
    master_iterator: Annotated[str, 'REFINE to RETRIVE Loop iterator']


# Create an agent node using the specified agent and name
def agent_node(state, agent, name):
    # Call the agent
    agent_response = agent.invoke(state)
    # Convert the last message of the agent to a HumanMessage and return it
    return {
        "messages": [
            HumanMessage(content=agent_response["messages"][-1].content, name=name)
        ]
    }



llm=AzureChatOpenAI(model="gpt-4o-mini", temperature=0,
                    api_key=model_api_key,
                    api_version="2024-02-15-preview",
                    azure_endpoint=model_openai_endpoint)
retriever_agent=create_react_agent(model=llm, tools=[retrieve_tool])
# retrieve_node=functools.partial(agent_node, agent=retriever_agent, name="retrieve_kb"
# )

def retrieve_node(state):

    print("============In Retrieve node==========\n")
    query = state["query"]
    hits = retrieve_tool.invoke({"query": query})  # invoke directly
    return {
        "kb_hits": hits["matches"],
        "messages": [
            HumanMessage(content=f"Retrieved {len(hits['matches'])} snippets", name="retrieve_kb")
        ]
    }





datum=pd.DataFrame(data)
# data[data['doc_id']=hit['id']]['answer_snippet']
def generate_answer_node(state):
    print("============In Generate Answer node==========\n")
    query = state["query"]

    kb_hits = state["kb_hits"]
    print(kb_hits)

    retrieved_snippets = "\n".join(
    f"{hit['id']} answer_snippet: {datum[datum['doc_id'] == hit['id']]['answer_snippet'].values[0]}"
    for hit in kb_hits if not datum[datum['doc_id'] == hit['id']].empty
)
    print(f"\n============\n{retrieved_snippets}")
    prompt_text = f"""You are a software best-practices assistant.

User Question:
{query}

Retrieved Snippets:
{retrieved_snippets}

Task:
Based on these snippets, write a concise answer to the user’s question.
Cite each snippet you use by its doc_id in square brackets (e.g., [KB004]).
Return only the answer text."""

    completion = llm.invoke(prompt_text)

    return {"kb_hits": kb_hits,
        "messages": [HumanMessage(content=completion.content, name="generate_answer")],
        "last_answer": completion.content,
        'generation_answer':completion.content
    }



def critique_answer_node(state):
  print("============In Crtique Answer node==========\n")
  query=state['query']
  intial_answer=state['last_answer']
  hits=state['kb_hits']
  retrieved_snippets = "\n".join(
    f"{hit['id']} answer_snippet: {datum[datum['doc_id'] == hit['id']]['answer_snippet'].values[0]}"
    for hit in hits if not datum[datum['doc_id'] == hit['id']].empty
)


  critique_prompt=f"""
        You are a critical QA assistant. The user asked: {query}

        Initial Answer:
        {intial_answer}

        KB Snippets:
        {retrieved_snippets}

        Task:
        Determine if the initial answer fully addresses the question using only these snippets.
        - If it does, respond exactly: COMPLETE
        - If it misses any point or cites missing info, respond: REFINE: <short list of missing topic keywords>

        Return exactly one line.
        """

  completion=llm.invoke(critique_prompt)
  print(f"STATUS: {completion.content}")
  return {
        "messages": [HumanMessage(content=completion.content, name="critique_answer")],
        "last_answer": completion.content,
        'kb_hits':hits,
        'critique_result':completion.content,
        'master_iterator':"first"
    }

def refine_answer_node(state):
  print("============In Refine Answer node==========\n")
  query=state['query']
  initial_answer=state['generation_answer']
  critique_result=state['last_answer']
  hits=state['kb_hits']
  master_iterator=state['master_iterator']
  print(f"RESULT: {critique_result.split(':')[0]},====,{critique_result.split(':')[1]}")

  if (critique_result.split(':')[0] is not None) and (master_iterator!='final_loop'):
    STATUS=critique_result.split(':')[0]
    refinement_query=critique_result.split(':')[1]
    return {
        'query':f"{query} and information on {refinement_query}",
        'master_iterator':"final_loop"
    }

  else:
    refine_prompt=f"""
          You are a software best-practices assistant refining your answer. The user asked: {query}

          Initial Answer:
          {initial_answer}

          Critique: {critique_result}

          Additional Snippet:
          [Code to display the single additional snippet’s doc_id and text]

          Task:
          Incorporate this snippet into the answer, covering the missing points.
          Cite any snippet you use by doc_id in square brackets.
          Return only the final refined answer."""

    completion=llm.invoke(refine_prompt)

    return {
          "messages": [HumanMessage(content=completion.content, name="refine_answer")],

          }
# generate_answer_agent=create_react_agent(model=llm, prompt=prompt)
# generate_answer_node=functools.partial(agent_node, agent=generate_answer_agent, name="generate_answer"
# )


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

def get_next(state):
    critique_result=state.get("critique_result","")
    if critique_result=="COMPLETE":
      return END
    elif critique_result.startswith("REFINE"):
      return "refine_answer"
    else:
      return END
    return state["next"]

workflow= StateGraph(AgentState)
workflow.add_node("retrieve_kb",retrieve_node)
workflow.add_edge(START, "retrieve_kb")
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("critique_answer", critique_answer_node)
workflow.add_node("refine_answer", refine_answer_node)

workflow.add_edge("retrieve_kb", "generate_answer")
workflow.add_edge("generate_answer", "critique_answer")
# workflow.add_edge("critique_answer", "refine_answer")
workflow.add_edge("refine_answer", "retrieve_kb")
workflow.add_conditional_edges(
    "critique_answer",
    get_next,
    {
        "refine_answer": "refine_answer",
        END: END
    }
)
graph = workflow.compile(checkpointer=MemorySaver())


inputs = {
    "query": "What are the best practices for coding?",
    "sender": "user",
}

# # Run it
# result = graph.invoke(inputs,config={"thread_id": "session-001"})
graph = workflow.compile()  # no checkpointer
result = graph.invoke(inputs)
print(result["messages"][-1].content)



# Output
# Embedding and Upserting: 100%|██████████████████████████████████████| 3/3 [00:07<00:00,  2.41s/it]
# ============In Retrieve node==========

# ============In Generate Answer node==========

# [{'id': 'KB006',
#  'metadata': {'last_updated': '2024-06-10', 'source': 'unit testing_guide.md'},
#  'score': 0.833226562,
#  'values': []}, {'id': 'KB026',
#  'metadata': {'last_updated': '2024-02-10', 'source': 'unit testing_guide.md'},
#  'score': 0.833226562,
#  'values': []}, {'id': 'KB016',
#  'metadata': {'last_updated': '2024-04-10', 'source': 'unit testing_guide.md'},
#  'score': 0.833226562,
#  'values': []}]

# ============
# KB006 answer_snippet: When addressing unit testing, it's important to follow well-defined patterns...
# KB026 answer_snippet: When addressing unit testing, it's important to follow well-defined patterns...
# KB016 answer_snippet: When addressing unit testing, it's important to follow well-defined patterns...
# ============In Crtique Answer node==========

# STATUS: REFINE: coding standards, version control, code reviews
# ============In Refine Answer node==========

# RESULT: REFINE,====, coding standards, version control, code reviews
# ============In Retrieve node==========

# ============In Generate Answer node==========

# [{'id': 'KB007',
#  'metadata': {'last_updated': '2024-01-10', 'source': 'CI/CD_guide.md'},
#  'score': 0.805420697,
#  'values': []}, {'id': 'KB017',
#  'metadata': {'last_updated': '2024-05-10', 'source': 'CI/CD_guide.md'},
#  'score': 0.805420697,
#  'values': []}, {'id': 'KB027',
#  'metadata': {'last_updated': '2024-03-10', 'source': 'CI/CD_guide.md'},
#  'score': 0.805420697,
#  'values': []}]

# ============
# KB007 answer_snippet: When addressing CI/CD, it's important to follow well-defined patterns...
# KB017 answer_snippet: When addressing CI/CD, it's important to follow well-defined patterns...
# KB027 answer_snippet: When addressing CI/CD, it's important to follow well-defined patterns...
# ============In Crtique Answer node==========

# STATUS: COMPLETE
# COMPLETE