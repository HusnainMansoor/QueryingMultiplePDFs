The installation of the chromadb library failed due to the lack of the necessary Microsoft Visual C++ Build Tools. However, the other dependencies were installed successfully. Here is a summary of the process and the code that was executed:

Installation of Required Libraries:

langchain, openai, chromadb (installation failed), pypdf, and tiktoken were installed to facilitate PDF summarization using the LangChain library.
Environment Setup:

The OpenAI API key was set as an environment variable.
Summarizing PDFs:

A function summarize_pdfs_from_folder was defined to iterate over PDF files in a specified folder, load and split the content of each PDF using PyPDFLoader, and then summarize the content using the load_summarize_chain function from LangChain with a map-reduce chain type.
A custom_summary function was also defined to generate summaries based on a custom prompt provided by the user.
Here is the code to summarize the PDFs:

python
Copy code
# Import necessary modules
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI, PromptTemplate
import glob

# Set up OpenAI API key
import os
os.environ["OPENAI_API_KEY"] = "Enter Key"
llm = OpenAI(temperature=0.2)

# Define function to summarize PDFs from a folder
def summarize_pdfs_from_folder(pdfs_folder):
    summaries = []
    for pdf_file in glob.glob(pdfs_folder + "/*.pdf"):
        loader = PyPDFLoader(pdf_file)
        docs = loader.load_and_split()
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
        print("Summary for: ", pdf_file)
        print(summary)
        print("\n")
        summaries.append(summary)
    return summaries

# Define function to generate custom summaries
def custom_summary(pdf_folder, custom_prompt):
    summaries = []
    for pdf_file in glob.glob("C:/Users/HusnainMansoor/Documents/pdf"):
        loader = PyPDFLoader(pdf_file)
        docs = loader.load_and_split()
        prompt_template = custom_prompt + """

        {text}

        SUMMARY:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="map_reduce", 
                                    map_prompt=PROMPT, combine_prompt=PROMPT)
        summary_output = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]
        summaries.append(summary_output)
    return summaries

# Install missing dependencies
!pip install pypdf
!pip install tiktoken

# Run the summarization function
summaries = summarize_pdfs_from_folder("C:/Users/HusnainMansoor/Documents/pdf")
Make sure to replace "Enter Key" with your actual OpenAI API key. This script will summarize all PDFs in the specified folder and print the summaries to the console. If you encounter further issues with chromadb or any other library, please let me know, and I can help troubleshoot them.
