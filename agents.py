import json
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

LLM_MODEL = "llama3"

# 1. Updated Graph State
class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    faithfulness_score: int  # New: Did it use only the context?
    relevance_score: int     # New: Did it answer the prompt?

def create_multi_agent_workflow(retriever):
    # We use format="json" for the evaluator to ensure we get a parseable response
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    llm_json = ChatOllama(model=LLM_MODEL, temperature=0.0, format="json")

    # ==========================================
    # Agent 1: The Researcher
    # ==========================================
    def researcher_node(state: RAGState):
        print("--- RESEARCHER: Searching Vector DB ---")
        docs = retriever.invoke(state["question"])
        formatted_context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return {"context": formatted_context}

    # ==========================================
    # Agent 2: The Synthesizer
    # ==========================================
    def synthesizer_node(state: RAGState):
        print("--- SYNTHESIZER: Drafting Response ---")
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful and precise assistant. Answer the user's question "
            "based ONLY on the provided context. If the answer is not in the context, "
            "state that you do not know.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        chain = prompt | llm
        response = chain.invoke({"context": state["context"], "question": state["question"]})
        return {"answer": response.content}

    # ==========================================
    # Agent 3: The Evaluator (NEW)
    # ==========================================
    def evaluator_node(state: RAGState):
        print("--- EVALUATOR: Grading Response ---")
        prompt = ChatPromptTemplate.from_template(
            "You are an impartial judge evaluating an AI assistant's response.\n"
            "Grade the following answer on two metrics from 0 to 100:\n"
            "1. faithfulness_score: Is the answer strictly based on the provided Context? (Make up nothing).\n"
            "2. relevance_score: Does the answer directly address the original Question?\n\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Answer: {answer}\n\n"
            "Output ONLY a valid JSON object with the keys 'faithfulness_score' and 'relevance_score'."
        )
        
        chain = prompt | llm_json
        response = chain.invoke({
            "context": state["context"], 
            "question": state["question"], 
            "answer": state["answer"]
        })
        
        # Parse the JSON output from the LLM
        try:
            scores = json.loads(response.content)
            f_score = scores.get("faithfulness_score", 0)
            r_score = scores.get("relevance_score", 0)
        except json.JSONDecodeError:
            print("Failed to parse evaluator JSON. Defaulting to 0.")
            f_score, r_score = 0, 0
            
        return {"faithfulness_score": f_score, "relevance_score": r_score}

    # ==========================================
    # Build the Graph
    # ==========================================

    # we use Stategraph for Deterministic Routing (LLM doesnt get to decide which agent to use), as well as for extensibility -
    #if we want a new feature we just add a new node (eg., "translator")
    workflow = StateGraph(RAGState)

    workflow.add_node("researcher", researcher_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("evaluator", evaluator_node)

    # Updated Flow: Researcher -> Synthesizer -> Evaluator -> END
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "synthesizer")
    workflow.add_edge("synthesizer", "evaluator")
    workflow.add_edge("evaluator", END)
    
    return workflow.compile()