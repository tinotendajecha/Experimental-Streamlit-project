import os
import time
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_community.vectorstores import Qdrant
import streamlit as st
from dotenv import load_dotenv
import time

# Read from .env file 
qdrant_host = os.getenv('QDRANT_HOST')

# Connect to the qdrant client
client = QdrantClient(url=qdrant_host)

# Function for querying vector points and retrieving their meta data
# def extract_meta_data(sources):
#     # Initialize empty lists
#     retrieved_docs = []
#     list_of_metadata = []
#     metadata = {}

#     # Loop the sources list and extract the id and collection name
#     for source in sources:
#         doc_id = source.metadata['_id']
#         collection_name = source.metadata['_collection_name']

#         # Query every document and append to list
#         document = client.retrieve(
#             collection_name=collection_name,
#             ids=[doc_id],
#             with_payload=True
#         )
        
#         # Append the retrieved docs to the list
#         retrieved_docs.append(document)
    
#     # Loop through the retrieved docs and extract the metadata
#     for each_doc in retrieved_docs:
#         record = each_doc[0]
        
#         # Destructuring the data type
#         retrieved_payload = record.payload

#         # Extract the metadata
#         source_doc = retrieved_payload['source']
#         page_content = retrieved_payload['page_content']
#         page_number = retrieved_payload['page']

#         # Create an object for each meta data
#         metadata = {
#             "Source Document" :source_doc,
#             "Page Content" :page_content,
#             "Page Number" : page_number
#         }

#         # Append the metadata to the list
#         list_of_metadata.append(metadata)

#     # Return the list of metadata
#     return list_of_metadata


# Function for retrieving the list of collections from qdrant store
def get_list_of_collections():

    # Fetch collections from qdrant store
    collections = client.get_collections()
    
    # Get the list of collections
    list_of_collections = collections.collections

    collections = [] #Initialize empty collection for appending collection_names

    # Loop through the list of collections and append the collection names
    for collection in list_of_collections:
        collections.append(collection.name)
    
    # Return the list of collections
    return collections


def main():
    # Load the environment variables
    load_dotenv()
    
    # Give chat app page a name / title
    st.set_page_config(page_title='ETF Docs')

    # Heading for the app
    st.header('ETF prospectus docs chat')

    # Get list of collections 
    collections = get_list_of_collections() # Invoke func for getting names of collections

    # Render a drop down displaying the list of collections to choose from
    vector_store_to_use = 'etf_docs_collection'
    
    # Drop down which displays a list with dimensions to use
    dimensions_to_use = 384
    
    # Grab the user question
    user_question = st.text_input("Prompt here!")

    # Check if the user has asked a question
    if user_question:

        # Configure the embedding model
        embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=dimensions_to_use)

        # Initialize the document store
        doc_store = Qdrant(
            client=client,
            collection_name= vector_store_to_use, # Can change the collection here 
            embeddings = embeddings_model
        )

        # Initialize the OpenAI model
        llm = OpenAI()

        # Initialize the retrieval QA chain
        qa = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type="stuff",
            retriever= doc_store.as_retriever(),
            return_source_documents=True
        )

        # Render the user question on the screen
        st.markdown(f':green[Question:] {user_question}')

        # Hook up the user question
        response = qa.invoke(user_question)

        # Print the response below
        st.markdown(':green[Response:]')
        st.write(response['result'])

        # time.sleep(1)

        # Print sources below
        # st.header(':red[Sources used:]')

        # sources = response['source_documents']

        # for source in sources:
        #     st.markdown(f':green[{source.page_content}]')

        # sources = response['source_documents']
        

        # call extract metadata here
        # list_of_metadata = extract_meta_data(sources)

        # # Loop through the list of metadata and display the source document, page content and page number
        # for payload in list_of_metadata:
        #     source_doc = payload['Source Document']
        #     page_content = payload['Page Content']
        #     page_number = payload['Page Number']

        #     st.markdown(f':green[Source Document:] {source_doc}')
        #     st.markdown(f':green[Page Number] {page_number}')
        #     st.markdown(f':green[Page Content:] {page_content}')
        #     st.divider()


if __name__ == '__main__':
    main()