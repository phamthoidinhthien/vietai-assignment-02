from utils.sheet_reader import load_data_from_google_spreadsheet
from utils.document_reader import document_loader
from utils.return_message import GradioAgentChatPack
from qe_manufacturer import get_custom_query_engine, get_custom_query_engine_have_filter
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.agent import AgentRunner
from llama_index.core.tools import QueryEngineTool
from llama_index.agent.openai import OpenAIAgentWorker
import tiktoken
import pandas as pd
import os

from llama_index.core import (
    VectorStoreIndex,
    load_index_from_storage,
    StorageContext
)

def initialize():
    
    # Settings
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=100)
    Settings.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini").encode
    token_counter = TokenCountingHandler()
    Settings.callback_manager = CallbackManager([token_counter])
    
    # Get the index, either by load from storage or refresh from data source
    # If exist, load from local storage
    print("Start loading index...")
    if os.path.exists("./db/default__vector_store.json"):
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./db"),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir="./db", namespace="default"),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir="./db"),
        )
        index = load_index_from_storage(storage_context, num_workers=4)
    
    # Else load from google spread sheet
    else:
        #load docs from spread sheet
        df = load_data_from_google_spreadsheet(spreadsheet_id="1PGG0gLU7DVC7omYJC9uJtInpw7V8uAPnPNjjGv9FSLc", worksheet_name="Events")  
        docs = document_loader(
                df, 
                metadata_columns=["Workgroup","Platform","PageGroup","Page","Category","Action","Project","Status"], 
                text_column= ["Description","Event name on GA4","Description on GA4","PageGroup","Page", "Category","Action","Label","Event CDs","Project","params describe action on GA4","value on GA4","other params on GA4"]
        )
        
        # get embedding, index to vector store index and persist to local storage
        index = VectorStoreIndex.from_documents(docs, show_progress=True, Settings=Settings)
        index.storage_context.persist("./db")
    print("Finish loading index")

    # Init query engine for search event
    search_tag = get_custom_query_engine(index=index, similarity_top_k=10)


    # Init query engine for search and filter by page
    if os.path.exists("./db/page_df.csv"):
        page_df = pd.read_csv("./db/page_df.csv")
    else:
        page_df = load_data_from_google_spreadsheet(spreadsheet_id="1PGG0gLU7DVC7omYJC9uJtInpw7V8uAPnPNjjGv9FSLc", worksheet_name="Pages")
        page_df.to_csv("./db/page_df.csv")
    
    # Preprocess the object before feed to query engine 
    page_objects = list(page_df["Page"].unique())
    page_objects= [x for x in page_objects if x!=""]

    # Init query engine for search event by page
    search_page = get_custom_query_engine_have_filter(index=index, similarity_top_k=10, filter_objects=page_objects)


    # Init tools for agent
    query_engine_tools = [
        QueryEngineTool.from_defaults(
            query_engine=search_tag,
            name="search_tag",
            description="Provides information about detail of event tracking data "
                "Use a detailed plain text question as input to the tool.",
        ),
        QueryEngineTool.from_defaults(
            query_engine=search_page,
            name="search_page",
            description="Provides information about tags base on pages' name. If user mention page, can use this tool. You must list all the tags of that page",
        ),
    ]

    # Init agent with agent worker and agent runner
    openai_step_engine = OpenAIAgentWorker.from_tools(query_engine_tools, llm=Settings.llm, verbose=True)
    agent = AgentRunner(openai_step_engine)

    return agent

if __name__ == "__main__":
    agent = initialize()
    gradio = GradioAgentChatPack(agent)
    gradio.run()