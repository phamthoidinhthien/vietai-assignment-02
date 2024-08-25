from llama_index.core import Document
import pandas as pd
from typing import List, Union

def document_loader(
    df: pd.DataFrame,
    text_column: List[str],
    metadata_columns: List[str],
    ) -> List[Document]:

    text_indices = [df.columns.get_loc(column) for column in text_column] 
    metadata_indices = [df.columns.get_loc(column) for column in metadata_columns]

    documents = []
    for row in df.itertuples(index=False):
        text = ""
        for column_name, column_index in zip(text_column, text_indices):
            value = 'None' if row[column_index] == '' else row[column_index]
            text += f"{column_name}: {value}; "
        
        metadata = {}
        for column_name, column_index in zip(metadata_columns, metadata_indices):
            metadata[column_name] =  row[column_index]
        
        document = Document(
                text=text,
                metadata=metadata,
                metadata_seperator="\n",
                metadata_template="{key}: {value}",
                text_template="Metadata: {metadata_str}\n----------\n\nContent: {content}",
            )
        
        documents.append(document)
        
    return documents