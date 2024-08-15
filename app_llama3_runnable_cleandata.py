import streamlit as st
from langchain.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser

# 1. Vectorise the sales response csv data
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata.update(record.get("metadata", {}))
    metadata["question"] = record.get("question")
    metadata["answer"] = record.get("answer")
    metadata["chunk_id"] = record.get("chunk_id")
    metadata["document_name"] = record.get("document_name")
    return metadata

loader = JSONLoader(
    file_path="qa_dataset_groundtruthJL.json",
    jq_schema='.[]',
    content_key="chunk_content",
    metadata_func=metadata_func
)

documents = loader.load()
embeddings = OllamaEmbeddings(model="llama3")  # "gemma:7b" "nomic-embed-text")
vectorstore = FAISS.from_documents(documents, embeddings)

# 2. Setup retriever
def filter_documents(documents, question):
    filtered_docs = [
        doc for doc in documents
        if "loan_amount" in doc.metadata and "loan_currency" in doc.metadata
    ]
    return filtered_docs

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3},
    filter_func=filter_documents
)

# 3. Setup LLM
llm = ChatOllama(model="llama3", temperature=0.7)

# 4. Prompt template
template = """
You are an AI-powered financial analyst specializing in loan agreements. Your task is to answer the following question about a financing agreement: {question}
Use the following information retrieved from the loan document to inform your answer:
{context}
Relevant metadata:
- Loan Amount: {loan_amount} {loan_currency}
- Loan Term: {loan_term}
- Key Information: {key_information}
Guidelines for your response:
1. Base your answer directly on the provided document content and metadata.
2. If you can't find the answer in the given information, say you don't know.
3. Cite specific sections or clauses when referencing the agreement (e.g., "As stated in Section 4...").
4. Provide a concise, well-structured answer that directly addresses the question.
Your response:
"""

prompt = ChatPromptTemplate.from_template(template)

# 5. Setup runnable chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_metadata(docs):
    if docs:
        metadata = docs[0].metadata
        return {
            "loan_amount": metadata.get("loan_amount", "N/A"),
            "loan_currency": metadata.get("loan_currency", "N/A"),
            "loan_term": metadata.get("loan_term", "N/A"),
            "key_information": metadata.get("key_information", "N/A")
        }
    return {
        "loan_amount": "N/A",
        "loan_currency": "N/A",
        "loan_term": "N/A",
        "key_information": "N/A"
    }

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        **({'loan_amount': RunnablePassthrough() | retriever | get_metadata | (lambda x: x['loan_amount']),
            'loan_currency': RunnablePassthrough() | retriever | get_metadata | (lambda x: x['loan_currency']),
            'loan_term': RunnablePassthrough() | retriever | get_metadata | (lambda x: x['loan_term']),
            'key_information': RunnablePassthrough() | retriever | get_metadata | (lambda x: x['key_information'])})
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Streamlit app
def main():
    st.set_page_config(page_title="Financing Agreement Analyzer", page_icon=":moneybag:")
    st.header("Financing Agreement Analyzer :moneybag:")

    with st.sidebar:
        st.write("Question Guide")
        st.markdown("""
        - **Loan Terms**: *"What are the key terms of the loan agreement?"*
        - **Repayment Schedule**: *"Can you explain the repayment schedule for this loan?"*
        - **Interest Rates**: *"What is the interest rate for this loan?"*
        - **Collateral**: *"What collateral or security has been provided for this loan?"*
        - **Covenants**: *"What are the main financial covenants in this agreement?"*
        - **Default Conditions**: *"Under what conditions would this loan be considered in default?"*
        """)

    if 'query_history' not in st.session_state:
        st.session_state['query_history'] = []

    message = st.text_area("Please type your question about the financing agreement here:")
    status_message = st.empty()

    if message:
        status_message.write("Generating response...")
        result = chain.invoke(message)
        status_message.write("Response Generated!")
        st.info(result)
        st.session_state['query_history'].append(message)

    st.write("Past Questions:")
    for past_query in st.session_state['query_history']:
        st.write(past_query)

if __name__ == "__main__":
    main()
