import pandas as pd
import openai
import json
import os
from docarray import BaseDoc
from docarray.typing import NdArray
from docarray import DocList
import numpy as np
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPEN_API_KEY')

class ToyDoc(BaseDoc):
  text: str = ''
  embedding: NdArray[1536]


def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   response = openai.Embedding.create(
     input = text,
     model = model
   )
   content = response['data'][0]['embedding']
   return content

def getListFromFile(fileName):
  file = open(fileName)
  return json.loads(file.read())

# t1 = "This is a test N1"
# td1 = ToyDoc(text=t1, embedding=get_embedding(t1))
# t2 = "This is a test N2"
# td2 = ToyDoc(text=t2, embedding=get_embedding(t2))
# t3 = "something irrelevant"
# td3 = ToyDoc(text=t3, embedding=get_embedding(t3))
# t4 = "This is a test N3"
# query = ToyDoc(text=t4, embedding=get_embedding(t4))

t1 = "This is a test N1"
td1 = ToyDoc(text=t1, embedding=getListFromFile('./vectors/c1.json'))

t2 = "This is a test N2"
td2 = ToyDoc(text=t2, embedding=getListFromFile('./vectors/c2.json'))

t3 = "something irrelevant"
td3 = ToyDoc(text=t3, embedding=getListFromFile('./vectors/c3.json'))

db = InMemoryExactNNVectorDB[ToyDoc](workspace='./workspace_path')
doc_list = [td3, td2, td1]
db.index(inputs=DocList[ToyDoc](doc_list))

queryText = "This is a test N3"
queryDoc = ToyDoc(text=queryText, embedding=getListFromFile('./vectors/c4.json'))

results = db.search(inputs=DocList[ToyDoc]([queryDoc]), limit=3)

for m in results[0].matches:
  print(m)
