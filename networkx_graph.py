
"""
Load Tamil Nadu government schemes into a NetworkX knowledge graph.

Flow:
schemes.json -> Documents -> Chunks -> LLMGraphTransformer -> NetworkX
"""

import json
import os
import pickle

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
import networkx as nx

load_dotenv()

# ---------------------------------------------------------------
# 1. Check environment variables
# ---------------------------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY in .env file")

print("OpenAI API key found.")

# ---------------------------------------------------------------
# 2. Load scheme records
# ---------------------------------------------------------------
with open("schemes.json", "r", encoding="utf-8") as file:
    schemes = json.load(file)

documents = []

for scheme in schemes:
    content = "\n".join(
        f"{key}: {value}"
        for key, value in scheme.items()
        if value is not None and str(value).strip()
    )

    documents.append(
        Document(
            page_content=content,
            metadata={
                "source": scheme.get(
                    "source_url", "Tamil Nadu Government"
                ),
                "scheme_name": scheme.get(
                    "scheme_name", "Unknown Scheme"
                ),
            },
        )
    )

print(f"Loaded {len(documents)} scheme documents.")

# ---------------------------------------------------------------
# 3. Split documents into chunks
# ---------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} text chunks.")

# ---------------------------------------------------------------
# 4. Initialize the LLM and graph transformer
# ---------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

transformer = LLMGraphTransformer(llm=llm)

# ---------------------------------------------------------------
# 5. Extract graph entities and relationships
# ---------------------------------------------------------------
graph_documents = []

# First, test only 3 chunks.
# After confirming success, change to: test_chunks = chunks
test_chunks = chunks


for i, chunk in enumerate(test_chunks, start=1):
    print(
        f"Processing chunk {i}/{len(test_chunks)}...",
        flush=True,
    )

    result = transformer.convert_to_graph_documents([chunk])
    graph_documents.extend(result)

    if result:
        print(
            f"Finished chunk {i}: "
            f"{len(result[0].nodes)} nodes, "
            f"{len(result[0].relationships)} relationships",
            flush=True,
        )

# ---------------------------------------------------------------
# 6. Build the NetworkX graph
# ---------------------------------------------------------------
G = nx.MultiDiGraph()

for graph_doc in graph_documents:
    # Add entity nodes
    for node in graph_doc.nodes:
        node_id = str(node.id)

        G.add_node(
            node_id,
            label=node.type,
            properties=node.properties or {},
        )

    # Add relationship edges
    for relationship in graph_doc.relationships:
        source_id = str(relationship.source.id)
        target_id = str(relationship.target.id)
        relation_type = str(relationship.type)

        # Ensure endpoints exist even if the transformer returns
        # a relationship endpoint without a separate node entry.
        if source_id not in G:
            G.add_node(
                source_id,
                label=relationship.source.type,
                properties={},
            )

        if target_id not in G:
            G.add_node(
                target_id,
                label=relationship.target.type,
                properties={},
            )

        G.add_edge(
            source_id,
            target_id,
            relation=relation_type,
        )

print("\nNetworkX graph created.")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Relationships: {G.number_of_edges()}")

# ---------------------------------------------------------------
# 7. Save the graph
# ---------------------------------------------------------------
output_file = "schemes_graph.gpickle"

with open(output_file, "wb") as file:
    pickle.dump(G, file)

print(f"\nGraph saved to {output_file}")

# ---------------------------------------------------------------
# 8. Display a sample of the graph
# ---------------------------------------------------------------
print("\nSample nodes:")
for node_id, attributes in list(G.nodes(data=True))[:10]:
    print(node_id, attributes)

print("\nSample relationships:")
for source, target, attributes in list(
    G.edges(data=True)
)[:10]:
    print(f"{source} --{attributes['relation']}--> {target}")

print("\nNetworkX knowledge graph is ready!")
