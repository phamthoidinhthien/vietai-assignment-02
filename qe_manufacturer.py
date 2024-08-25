from llama_index.core import get_response_synthesizer, PromptTemplate, QueryBundle, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterCondition
from llama_index.core.retrievers import QueryFusionRetriever, BaseRetriever
from llama_index.core.objects import ObjectIndex
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.schema import NodeWithScore
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.retrievers.bm25 import BM25Retriever
from typing import List
    
class CustomRetriever(BaseRetriever):
    """Custom retriever that performs search for filter before apply filter"""

    def __init__(
        self,
        object_retriever: BaseRetriever,
        index: VectorStoreIndex,
        similarity_top_k: int = 10,
    ) -> None:
        
        """Init params."""
        self._object_retriever = object_retriever
        self._index = index
        self._similarity_top_k = similarity_top_k
        # self._keyword_retriever = keyword_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        object_nodes = self._object_retriever.retrieve(query_bundle)
        filtersmap = [MetadataFilter(key="Page", value=value) for value in object_nodes]
        filters = MetadataFilters(
            filters=filtersmap,
            condition=FilterCondition.OR,
        )
        retrieve_nodes = self._index.as_retriever(similarity_top_k=self._similarity_top_k, filters=filters).retrieve(query_bundle)
    
        return retrieve_nodes


class RAGQueryEngine(CustomQueryEngine):
    """RAG Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        response_obj = self.response_synthesizer.synthesize(query_str, nodes)
        return response_obj


def get_custom_response_synthesizer() -> BaseSynthesizer:
    """Get response synthesizer."""
    # Prompt for response synthesizer
    qa_prompt_tmpl_str = """\
    You are a helper that return answer of how event is tag/tracked in the data system
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, \
    answer the query.
    Please provide your answer in the form of a structured containing \
    the page of the event, you must list event description, event category, action, label, possible Events CD (Custom dimension), and project if available. Don't use markdown bold **. Some examples are given as below:

    Page: Adview - Contact seller
    Description: when user click to view personal page when contact the seller
    Category: contact_seller
    Action: click
    Label: view_personal_page
    Some additional CDs in the event: ad_id and category_id
    Project: User Profile

    If you found multiple events that match user query, please list all of them, remember it must answer the user's query, if not don't list it
    Query: {query_str}
    Answer: \
    """

    # Construct the response synthesizer
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    response_synthesizer = get_response_synthesizer()
    response_synthesizer.update_prompts(
        {"text_qa_template": qa_prompt_tmpl}
    )

    return response_synthesizer

def get_custom_query_engine(
        index: VectorStoreIndex,
        similarity_top_k: int = 10,
    ) -> CustomQueryEngine:

    # Prompt for generate more queries
    query_gen_str = """\
    You are a helpful assistant that generates multiple search queries based on a \
    single input query. User is trying to search for code tag tracking on a classified platform \
    If the question in vietnamese, translate it to english before generate the queries. \
    You can permuate the queries using these rule, it can be combine together: 
    - use synonym words to permuate the tag name/word/object, for example: reco can be recommendation, personalize; purchase can be buy, all services can be premium feature; lead can be sms, call or chat etc. Don't change the meaning of it
    - use _ to connect words for the event name, don't connect all word in the question, only event name , for example: gds subscription can also be gds_subscription

    Generate {num_queries} search queries, one on each line, related to the following input query, you must put original query as first item in the list:
    Original Query: {query}
    Queries list:
    """

    # Construct the retriever
    retriever = QueryFusionRetriever(
        [
            index.as_retriever(
                similarity_top_k=5, 
            ),
            
            BM25Retriever.from_defaults(
                docstore=index.docstore, 
                similarity_top_k=15,
            ),
        ],
        
        num_queries=5,
        mode= "simple",
        similarity_top_k=similarity_top_k,
        use_async=True,
        verbose=True,
        query_gen_prompt=query_gen_str,
        retriever_weights=[0.4, 0.6],    
    )
    
    # Construct the response synthesizer
    response_synthesizer = get_custom_response_synthesizer()

    # Construct the query engine
    query_engine = RAGQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

    return query_engine


def get_custom_query_engine_have_filter(
        index: VectorStoreIndex,
        filter_objects: List[str],
        similarity_top_k: int = 10,
    ) -> CustomQueryEngine:

    # Build object retrieve to get item for filter
    object_index = ObjectIndex.from_objects(
        filter_objects, index_cls=VectorStoreIndex
    )
    object_retriever = object_index.as_retriever(similarity_top_k=4)


    custom_retriever_with_filter = CustomRetriever(object_retriever=object_retriever, index=index, similarity_top_k=similarity_top_k)
    custom_response_synthesizer = get_custom_response_synthesizer()
    query_engine = RAGQueryEngine(retriever=custom_retriever_with_filter, response_synthesizer=custom_response_synthesizer, similarity_top_k=similarity_top_k)

    return query_engine
