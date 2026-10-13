# Tamil Nadu Government Schemes: Graph RAG Chatbot

A chatbot that answers questions about Tamil Nadu government schemes using a **knowledge graph** instead of plain text search. Scheme data is converted into entities and relationships with an LLM, stored as a graph, and used to ground the chatbot's answers.

The graph is built two ways:

- **Neo4j AuraDB**: a cloud graph database that can be queried with Cypher and visualised.
- **NetworkX**: a lightweight local graph (no server needed) that powers the chatbot.

## How it works

```
schemes.json -> Documents -> Chunks -> LLMGraphTransformer (GPT-4o-mini) -> Graph
                                                                              |
                                          +-----------------------------------+
                                          |                                   |
                                    Neo4j AuraDB                    NetworkX (.gpickle)
                                 (query + visualise)                         |
                                                                       rag_chatbot.py
```

**Chatbot flow (`rag_chatbot.py`):**

1. The LLM extracts search concepts from the user's question.
2. The script finds matching nodes in the NetworkX graph and gathers their connected neighbours.
3. The LLM writes an answer using only that retrieved graph context.

## Project structure

| File | Purpose |
|---|---|
| `collect_schemes.py` | Collects scheme data from government sources into `schemes.json` |
| `validate_schemes.py` | Validates the collected scheme records |
| `schemes.json` | Source data (scheme records) |
| `scrape_failures.json` | Log of pages that failed during collection |
| `load_to_neo4j.py` | Extracts the graph with an LLM and loads it into Neo4j AuraDB |
| `networkx_graph.py` | Extracts the graph with an LLM and saves it as `schemes_graph.gpickle` |
| `rag_chatbot.py` | Interactive Graph RAG chatbot using the NetworkX graph |

## Tech stack

Python, LangChain, OpenAI (`gpt-4o-mini`), Neo4j AuraDB, NetworkX

## Setup

### 1. Create a virtual environment and install packages

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install langchain-core langchain-openai langchain-text-splitters \
            langchain-experimental langchain-neo4j neo4j networkx \
            python-dotenv certifi truststore
```

### 2. Create a Neo4j AuraDB instance (for the Neo4j part)

1. Go to <https://console.neo4j.io> and create a **Free** instance.
2. Download the credentials file when shown. The password is displayed **only once**.
3. Copy the **Connection URI** exactly as shown, for example `neo4j+s://xxxxxxxx.databases.neo4j.io`.

### 3. Create a `.env` file

```
OPENAI_API_KEY=your-openai-key
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-aura-password
NEO4J_DATABASE=xxxxxxxx
```

`NEO4J_DATABASE` is the instance ID (the first part of the URI) on newer Aura instances. Never commit this file.

## Usage

**Build the Neo4j knowledge graph:**

```bash
python load_to_neo4j.py
```

**Build the NetworkX graph and run the chatbot:**

```bash
python networkx_graph.py
python rag_chatbot.py
```

By default both build scripts process only the first 3 chunks as a quick test:

```python
test_chunks = chunks[:3]
```

To process all the data, change it to:

```python
test_chunks = chunks
```

and run the build script again. The chatbot reads the saved `schemes_graph.gpickle`, so **rebuild the graph after any change**.

### Visualising in Neo4j

Open your instance in the Aura console and use Neo4j Browser / Explore to see how schemes, departments, beneficiaries and benefits connect. A useful starting query:

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50
```

## Example questions

- Tell me about the Quality Paddy seed Distribution scheme
- What is the funding pattern for paddy seeds?
- What schemes are available for farmers?

Ask complete questions. The chatbot has no memory yet, so follow-ups like "Who is eligible for this scheme?" will not work.

## Troubleshooting

| Error | Cause and fix |
|---|---|
| `SSLCertVerificationError: self-signed certificate in certificate chain` | A network proxy or antivirus is intercepting HTTPS. Install `truststore` and add `import truststore; truststore.inject_into_ssl()` at the very top of the script. |
| `DatabaseNotFound: database 'neo4j' does not exist` | Newer Aura instances use the instance ID as the database name. Set `NEO4J_DATABASE` in `.env` and pass `database=` to `Neo4jGraph`. |
| Chatbot says it has no information | The graph is probably too small. Check that `test_chunks = chunks`, then rebuild the graph. |
| Authentication error | Wrong Aura password. Reset it in the console and update `.env`. |

## What I learnt

- **Read connection errors carefully.** The SSL error came from my Windows network, not from Neo4j.
- **A chatbot is only as good as its graph.** It said there were no paddy schemes because only 3 of 93 chunks had been processed.
- **Field names must match the data.** The data uses `Scheme Title/Name`, but the code looked for `scheme_name`. Adding the scheme name to every chunk lets details such as eligibility and funding link to the right scheme.
- **Visualising the graph helps check the data**, because gaps and missing links are easy to spot.
- **Protect secrets.** `.env` and credential files are excluded with `.gitignore`.

## Known limitations

- No conversation memory.
- Retrieval depends on LLM-extracted concepts matching node names.
- `langchain-experimental` is being sunset, so the graph transformer may need replacing later.

## Future work

- Add chat history for follow-up questions.
- Combine vector search with graph retrieval.
- Fix scraped fields (for example, missing department names and currency symbols).
- Add a simple web interface.