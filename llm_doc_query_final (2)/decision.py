from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

def evaluate_with_llm(query, vectorstore):
    prompt_template = '''
You are an insurance policy assistant. Answer the question based on the document.

Question: {question}

If the question is outside the document, respond with "Not mentioned in the policy."
'''
    prompt = PromptTemplate.from_template(prompt_template)

    groq_llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=groq_llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    result = qa_chain({"query": query})
    return {"justification": result["result"]}