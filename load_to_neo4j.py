"""
Load Tamil Nadu scheme data into a Neo4j AuraDB knowledge graph.

Flow:
  schemes.json -> Documents -> chunks -> LLMGraphTransformer (OpenAI) -> Neo4j AuraDB
"""

# ---------------------------------------------------------------
# SSL FIX (must stay at the very top, before other imports)
# ---------------------------------------------------------------
# Preferred: use the Windows certificate store (works behind office
# proxies / antivirus). Needs:  pip install truststore
# Fallback: use certifi's CA bundle.
import os
import certifi

try:
    import truststore
    truststore.inject_into_ssl()
    print("SSL: using Windows certificate store (truststore)")
except ImportError:
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    print("SSL: using certifi bundle:", certifi.where())

import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph

load_dotenv()

# ---------------------------------------------------------------
# 1. Check environment variables
# ---------------------------------------------------------------
for var in ["OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]:
    if not os.getenv(var):
        raise ValueError(f"Missing {var} in .env file")

print("URI in use:", os.environ["NEO4J_URI"])

# ---------------------------------------------------------------
# 2. Test the Neo4j connection FIRST (before paying for OpenAI calls)
# ---------------------------------------------------------------
graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    database=os.environ.get("NEO4J_DATABASE", "neo4j"),
)
print("Connected to Neo4j!")

# ---------------------------------------------------------------
# 3. Load scheme records
# ---------------------------------------------------------------
with open("schemes.json", "r", encoding="utf-8") as file:
    schemes = json.load(file)

documents = []

for scheme in schemes:
    # Convert each structured scheme record into readable text
    content = "\n".join(
        f"{key}: {value}"
        for key, value in scheme.items()
        if value is not None and str(value).strip()
    )

    documents.append(
        Document(
            page_content=content,
            metadata={
                "source": scheme.get("source_url", "Tamil Nadu Government"),
                "scheme_name": scheme.get("scheme_name", "Unknown Scheme"),
            },
        )
    )

print(f"Loaded {len(documents)} scheme documents.")

# ---------------------------------------------------------------
# 4. Split documents into chunks
# ---------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} text chunks.")

# ---------------------------------------------------------------
# 5. Initialize the LLM
# ---------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# ---------------------------------------------------------------
# 6. Extract graph entities and relationships
# ---------------------------------------------------------------
transformer = LLMGraphTransformer(llm=llm)

graph_documents = []

# Quick test: only the first 3 chunks.
# To process everything, change to:  test_chunks = chunks
test_chunks = chunks[:3]

for i, chunk in enumerate(test_chunks, start=1):
    print(f"Processing chunk {i}/{len(test_chunks)}...", flush=True)

    result = transformer.convert_to_graph_documents([chunk])
    graph_documents.extend(result)

    print(
        f"Finished chunk {i}: "
        f"{len(result[0].nodes) if result else 0} nodes extracted",
        flush=True,
    )

print(f"Extracted {len(graph_documents)} graph documents.")

total_nodes = sum(len(doc.nodes) for doc in graph_documents)
total_relationships = sum(len(doc.relationships) for doc in graph_documents)

print(f"Extracted nodes: {total_nodes}")
print(f"Extracted relationships: {total_relationships}")

# ---------------------------------------------------------------
# 7. Store the extracted graph in AuraDB
# ---------------------------------------------------------------
graph.add_graph_documents(
    graph_documents,
    include_source=True,
    baseEntityLabel=True,
)

print("\nGraph Transformer import completed!")

# ---------------------------------------------------------------
# 8. Display graph counts
# ---------------------------------------------------------------
counts = graph.query("""
MATCH (n)
RETURN labels(n) AS node_labels, count(*) AS total
ORDER BY total DESC
""")

print("\nNode counts by label:")
for row in counts:
    print(row)

print("\nKnowledge graph is ready in Neo4j AuraDB.")