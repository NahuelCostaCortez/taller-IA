import gradio as gr

from langchain.document_loaders import OnlinePDFLoader

from langchain.text_splitter import CharacterTextSplitter

from langchain.llms import HuggingFaceHub

from langchain.embeddings import HuggingFaceHubEmbeddings

from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA

import os



def loading_pdf():
    return "Loading..."

def pdf_changes(pdf_doc, repo_id, API):

    if pdf_doc is None:
        return "Por favor, cargue un pdf"

    if API == "":
        return "Por favor, inserta tu API KEY de HuggingFace"
    
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = API
    loader = OnlinePDFLoader(pdf_doc.name)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceHubEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.1, "max_new_tokens":250})
    global qa 
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return "Ready"

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    # check if qa is defined
    if history[-1][0] == "":
        history[-1][1] = "Por favor, cargue un pdf o escriba una pregunta"
        return history
    response = infer(history[-1][0])
    history[-1][1] = response['result']
    return history

def infer(question):
    
    query = question
    result = qa({"query": query})

    return result

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with PDF</h1>
    <p style="text-align: center;">Sube un .PDF desde tu ordenador, haz clic en el botón "Cargar PDF en LangChain", <br />
    cuando todo esté listo, puedes empezar a hacer preguntas sobre el pdf ;)</p>
</div>
"""


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
        
        with gr.Column():
            pdf_doc = gr.File(label="Cargar pdf", file_types=['.pdf'], type="file")
            API = gr.Textbox(label="API KEY", placeholder="Inserta tu API KEY de HuggingFace ")
            repo_id = gr.Dropdown(label="LLM", choices=["OpenAssistant/oasst-sft-1-pythia-12b", "google/flan-ul2", "bigscience/bloomz"], value="OpenAssistant/oasst-sft-1-pythia-12b")
            with gr.Row():
                langchain_status = gr.Textbox(label="Status", placeholder="", interactive=False)
                load_pdf = gr.Button("Cargar pdf")
        
        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=350)
        question = gr.Textbox(label="Pregunta", placeholder="Escriba su pregunta y pulse Intro ")
        submit_btn = gr.Button("Enviar")
    #load_pdf.click(loading_pdf, None, langchain_status, queue=False)    
    repo_id.change(pdf_changes, inputs=[pdf_doc, repo_id], outputs=[langchain_status], queue=False)
    load_pdf.click(pdf_changes, inputs=[pdf_doc, repo_id, API], outputs=[langchain_status], queue=False)
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )
    submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )

demo.launch(share=True)