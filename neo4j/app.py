from langchain.graphs import Neo4jGraph
import os 


url = "neo4j+s://databases.neo4j.io"
username ="neo4j"

password = os.environ.get("NEO4J_PASSWORD")
graph = Neo4jGraph(
   url=url,
   username=username,
   password=password
)
