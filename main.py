from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import WikipediaLoader
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import warnings
warnings.filterwarnings("ignore")

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyB37wb6zM1sfeJMuGW_vOVCeTDcauZ05HY"

print("imports loaded...")
diffbot_api_key = "a704bada0b6aa1790514d11d00d8bd74"
diffbot_nlp = DiffbotGraphTransformer(diffbot_api_key=diffbot_api_key)
print("diffbot loaded...")
query = "Warren Buffett"
print("downloading data...")
raw_documents = WikipediaLoader(query=query).load()
print("warren buffet document loaded...")
print("converting to graph documents...")
graph_documents = diffbot_nlp.convert_to_graph_documents(raw_documents)
url = "neo4j+s://b6d6813e.databases.neo4j.io"
username = "neo4j"
password = "EKXfvO5ZbzKX6fJ0fv-r9QSjETwe53aLIWt913gdda0"
print("graph documents loaded!!!")
graph = Neo4jGraph(url=url, username=username, password=password)
graph.add_graph_documents(graph_documents)
graph.refresh_schema()
print("graph loaded!!!")


def load(inp):
    print("query recieved!!!")
    print("running chain...")
    chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0),
    qa_llm=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0),
    graph=graph,
    verbose=True,
    )
    #print("answer retrieved")
    return chain.run(inp)

for i in range(5):
    x=input("Enter your query: ")
    print(load(x))


