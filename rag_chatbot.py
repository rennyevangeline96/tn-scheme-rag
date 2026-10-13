
"""
Tamil Nadu Government Schemes - NetworkX Graph RAG Chatbot

Flow:
User question -> LLM extracts search concepts -> NetworkX graph retrieval
-> Retrieved context -> OpenAI -> Answer
"""

import json
import os
import pickle
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import networkx as nx


# ---------------------------------------------------------------
# 1. Load environment variables and graph file
# ---------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
GRAPH_FILE = BASE_DIR / "schemes_graph.gpickle"

load_dotenv(BASE_DIR / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY in .env file")

if not GRAPH_FILE.exists():
    raise FileNotFoundError(
        f"{GRAPH_FILE.name} not found. Run networkx_graph.py first."
    )

# Only load pickle files that you created and trust.
with GRAPH_FILE.open("rb") as file:
    G = pickle.load(file)

if not isinstance(G, nx.Graph):
    raise TypeError("The saved file does not contain a NetworkX graph.")

print("NetworkX knowledge graph loaded!")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Relationships: {G.number_of_edges()}")


# ---------------------------------------------------------------
# 2. Initialize OpenAI
# ---------------------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=60,
    max_retries=2,
)


# ---------------------------------------------------------------
# 3. Extract concepts from the user's question
# ---------------------------------------------------------------

concept_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract important search concepts from the user's question "
            "for a Tamil Nadu government schemes knowledge graph. Include "
            "scheme names, departments, beneficiary groups, benefits, and "
            "eligibility concepts where relevant. Return ONLY a JSON array "
            "of short strings. Do not answer the question."
        ),
        ("human", "{question}"),
    ]
)


def extract_concepts(question):
    response = (concept_prompt | llm).invoke({"question": question})
    content = response.content

    if isinstance(content, list):
        content = " ".join(
            str(item.get("text", "")) if isinstance(item, dict)
            else str(item)
            for item in content
        )

    content = str(content).strip()
    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE,
    )

    try:
        concepts = json.loads(content)
        if isinstance(concepts, list):
            return [
                str(concept).strip()
                for concept in concepts
                if str(concept).strip()
            ]
    except json.JSONDecodeError:
        pass

    # Fallback if the LLM does not return valid JSON.
    return [
        line.strip(" -*\"'")
        for line in content.splitlines()
        if line.strip()
    ][:8]


# ---------------------------------------------------------------
# 4. Search NetworkX nodes and retrieve connected graph facts
# ---------------------------------------------------------------

def node_text(node_id):
    attributes = G.nodes[node_id]
    properties = attributes.get("properties", {})

    return (
        str(node_id)
        + " "
        + str(attributes.get("label", ""))
        + " "
        + json.dumps(properties, ensure_ascii=False, default=str)
    )


def retrieve_graph_context(question, max_nodes=15):
    concepts = extract_concepts(question)

    print(
        "  Search concepts:",
        ", ".join(concepts) if concepts else "(none extracted)",
    )

    node_texts = {
        node_id: node_text(node_id)
        for node_id in G.nodes
    }

    # Match graph nodes to the concepts extracted by the LLM.
    matched_nodes = []
    seen = set()

    for concept in concepts:
        concept_lower = concept.casefold()
        words = {
            word
            for word in re.findall(r"\w+", concept_lower)
            if len(word) > 2
        }

        scores = []

        for node_id, text in node_texts.items():
            text_lower = text.casefold()
            score = 0

            if concept_lower and concept_lower in text_lower:
                score += 5

            score += sum(
                1 for word in words if word in text_lower
            )

            if score > 0:
                scores.append((score, node_id))

        scores.sort(key=lambda item: item[0], reverse=True)

        for _, node_id in scores[:4]:
            if node_id not in seen:
                matched_nodes.append(node_id)
                seen.add(node_id)

    # Fallback search if no LLM-generated concept matches.
    if not matched_nodes:
        stop_words = {
            "what", "which", "who", "when", "where", "why", "how",
            "does", "the", "for", "and", "with", "from", "tell",
            "about", "can", "scheme", "schemes",
        }

        words = [
            word
            for word in re.findall(r"\w+", question.casefold())
            if len(word) > 2 and word not in stop_words
        ]

        scores = []

        for node_id, text in node_texts.items():
            score = sum(
                1 for word in set(words)
                if word in text.casefold()
            )
            if score:
                scores.append((score, node_id))

        scores.sort(key=lambda item: item[0], reverse=True)
        matched_nodes = [node_id for _, node_id in scores[:6]]

    if not matched_nodes:
        return "", 0

    # Expand retrieval to include one-hop neighbors.
    selected_nodes = set(matched_nodes)

    for node_id in matched_nodes:
        selected_nodes.update(G.successors(node_id))
        selected_nodes.update(G.predecessors(node_id))

    # Keep direct matches first, followed by connected nodes.
    ordered_nodes = list(dict.fromkeys(
        matched_nodes
        + [node for node in selected_nodes if node not in matched_nodes]
    ))[:max_nodes]

    allowed_nodes = set(ordered_nodes)
    facts = []

    for node_id in ordered_nodes:
        attributes = G.nodes[node_id]

        facts.append(
            "ENTITY: "
            + str(node_id)
            + " | TYPE: "
            + str(attributes.get("label", "Entity"))
            + " | PROPERTIES: "
            + json.dumps(
                attributes.get("properties", {}),
                ensure_ascii=False,
                default=str,
            )
        )

        # Include outgoing relationships.
        for _, target, edge_attributes in G.out_edges(
            node_id, data=True
        ):
            if target in allowed_nodes:
                relation = edge_attributes.get(
                    "relation", "RELATED_TO"
                )
                facts.append(
                    f"RELATIONSHIP: {node_id} "
                    f"--[{relation}]--> {target}"
                )

        # Include incoming relationships.
        for source, _, edge_attributes in G.in_edges(
            node_id, data=True
        ):
            if source in allowed_nodes:
                relation = edge_attributes.get(
                    "relation", "RELATED_TO"
                )
                facts.append(
                    f"RELATIONSHIP: {source} "
                    f"--[{relation}]--> {node_id}"
                )

    # Remove repeated facts while preserving order.
    facts = list(dict.fromkeys(facts))

    return "\n".join(facts), len(ordered_nodes)


# ---------------------------------------------------------------
# 5. Generate an answer using the retrieved graph context
# ---------------------------------------------------------------

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant answering questions about Tamil Nadu "
            "government schemes. Use only the supplied knowledge graph "
            "context. Do not invent scheme names, benefits, eligibility "
            "criteria, amounts, deadlines, or application procedures. "
            "If the context is insufficient, clearly say that the current "
            "knowledge graph does not contain enough information. "
            "Give concise, helpful answers."
        ),
        (
            "human",
            "Knowledge graph context:\n{context}\n\n"
            "User question: {question}\n\n"
            "Answer:",
        ),
    ]
)

answer_chain = answer_prompt | llm


def ask_question(question):
    context, node_count = retrieve_graph_context(question)

    print(f"  Retrieved graph nodes: {node_count}")

    if not context:
        return (
            "I couldn't find relevant information in the current "
            "knowledge graph."
        )

    response = answer_chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    content = response.content

    if isinstance(content, list):
        content = " ".join(
            str(item.get("text", "")) if isinstance(item, dict)
            else str(item)
            for item in content
        )

    return str(content)


# ---------------------------------------------------------------
# 6. Interactive RAG terminal
# ---------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print(" Tamil Nadu Government Schemes - Graph RAG")
    print(" NetworkX + LangChain + OpenAI")
    print("=" * 60)
    print("Ask questions about Tamil Nadu government schemes.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.casefold() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        try:
            answer = ask_question(question)
            print(f"\nAssistant: {answer}\n")
        except Exception as error:
            print(f"\nError while answering: {error}\n")


if __name__ == "__main__":
    main()
