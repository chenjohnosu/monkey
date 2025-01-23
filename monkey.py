py# monkey v 0.5 Init
#     Johnny's academic research tool to ingest data/notes/articles and allow interactive query.
#
#     "If you give enough monkeys enough typewriters, they will eventually type out the
#     complete works of Shakespeare." -- Émile Borel (1913)
#
#     This is a general tool that will take a directory full of PDFs and docx
#     files, convert them to .txt files, then create a vector database for
#     RAG use with multiple LLMs.
#
#     Johnny Chen
#     College of Business
#     Oregon State University
#     Center for Marketing & Consumer Insights
#     chenjohn@oregonstate.edu
#     10/28/2024
#
# INSTALLATION: Requires local Ollama
#
# 4 Modes
# --grind               Generate embeddings/vector DB (first pass; required)
# --wrench              Query Vector Database
#       --do            Single question on Command line
#       * default       Interactive Q&A mode
# --pmode               Experimental: SmartDataFrame / Interactive pandasai
# --merge               Expand vdb by merging new source to target vdb store
#
# Helper tools:         chimp.py
#                       If PDF is image based, use tesseract OCR to extract text
#                       Run chimp separately on a directory of image based PDFs then
#                       copy into biz directory or grind first then merge
#
# Dev Log:
# 0.4 Moved index as query engine to CitationQueryEngine; configured "--kn"
# 0.5 Added merge mode of two organs to grow db without having to re-index/embed
#     More error checking and code cleanup
#
# NEW: Add all types of text data available for txt, docx, and pdfs; will ingest
#      .md, .epub, and .mbox but no error checking!
#
#
#
import argparse
import os
import sys
import PyPDF2
import docx2txt
import textwrap
import time
import re
#
#
import pandas as pd
from pandasai.smart_dataframe import SmartDataframe
#
# Llama
from llama_index.core import (VectorStoreIndex,
                              SimpleDirectoryReader,
                              Settings,
                              StorageContext,
                              load_index_from_storage)
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from langchain_ollama import OllamaLLM as LangchainOllama
from llama_index.core.node_parser import SentenceSplitter
import torch
#
default_src = "src"
default_vdb = "vdb"
default_llm = "mistral"
default_temp = 0.7
default_k = 5
line_width = 80
#
Settings.embed_model = HuggingFaceEmbedding(
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    device='cuda'
)
#
Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)  # default; future config
#
# Future Raw_Prompt instructions
# GUIDE = ("You are an academic research assistant helping to write academic journal papers. \
# Be verbose but paraphrase and do not copy text directly from the articles; \
# Do not make up citations; include inline citations but do not include the citations from the article content;\
# do not include references in the response.")
GUIDE = ("You are a research assistant helping analyze a set of interviews using thematic analysis.")


# "You are a lifestyle blogger and botanist writing for a website"
#
#  END CONFIG


def clean_text(text):
    # """Clean up extracted text from a PDF."""
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;!?]', '', text)
    return text.strip()


def txt_extension_check(filename):
    root, ext = os.path.splitext(filename)
    if os.path.exists(filename) and os.path.exists(root+'.txt'):
        print(f"Skipping {root}.txt exists")
        return True
    else:
        return False


def convert_files_to_text(directory):
    # Iterate through all files and subdirectories in the specified directory
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)

            # Ignore converting all .txt file -- already ready
            if not filename.endswith('.txt'):

                # Check if the file is a PDF
                if filename.endswith('.pdf') and not txt_extension_check(file_path):
                    print(f"Converting {filename}.")
                    with open(file_path, 'rb') as pdf_file:
                        reader = PyPDF2.PdfReader(pdf_file)
                        text = ''
                        for page_num in range(len(reader.pages)):
                            page = reader.pages[page_num]
                            text += page.extract_text()

                        # Clean up PDF
                        text = clean_text(text)
                        # Write the extracted text to a .txt file
                        with open(file_path.replace('.pdf', '.txt'), 'w', encoding='utf-8') as text_file:
                            text_file.write(text)

                # Check if the file is a DOCX
                elif filename.endswith('.docx') and not txt_extension_check(file_path):
                    print(f"Converting {filename}.")
                    text = docx2txt.process(file_path)

                    text = clean_text(text)

                    # Write the extracted text to a .txt file
                    with open(file_path.replace('.docx', '.txt'), 'w', encoding='utf-8') as text_file:
                        text_file.write(text)


def get_meta(file_path):
    return {"file_path": file_path}


def pmode(filename, model):
    print("Data Analysis Mode:\t\t" + filename)
    print("LLM Model:\t\t\t" + model + "\n")
    print("Professor Monkey is in the room!  Say 'quit' to exit")
    data = pd.read_csv(filename)
    llm = LangchainOllama(model=model)
    try:
        df = SmartDataframe(data, config={"llm": llm, "conversational": True})
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(False)
    while True:
        user_input = input("\nQuestion: ")
        if user_input.lower() == 'quit':
            exit(True)
        prompt = GUIDE + user_input
        response = df.chat(prompt)
        print(f"Answer:\n {response}")


def main():
    parser = argparse.ArgumentParser(prog="monkey", description='Usage for monkey business. \
        Research RAG for pdf, txt, and docx (and data).')
    parser.add_argument('-b', '--biz', type=str, help='Directory of documents: (optional, default: src)')
    parser.add_argument('-d', '--do', type=str, help='Single Query to the documents')
    parser.add_argument('-g', '--grind', action="store_true", help='MODE: Create vector store with documents')
    parser.add_argument('-k', '--knoodles', type=int, help='Manually configure k-retrieved (optional, default: 5)')
    parser.add_argument('-m', '--merge', type=str, help='MODE: Merge identified vdb to main organ (optional)')
    parser.add_argument('-p', '--pmode', type=str, help='MODE: Data Mode (csv file) -- Interactive only')
    parser.add_argument('-o', '--organ', type=str, help='Vector store name (optional, default: vdb)')
    parser.add_argument('-s', '--see', type=str, help='Use LLM Model (optional, default: mistral)')
    parser.add_argument('-t', '--temp', type=float, help='Set LLM temperature (optional, default: 0.7)')
    parser.add_argument('-w', '--wrench', action="store_true", help='MODE: Load vector store to query')
    parser.add_argument('-i', '--impress', type=str, help='Do impression of (GUIDE)')
    parser.add_argument('-v', '--verbose', action="store_true", help='Show sources and detailed debug information.')

    args = parser.parse_args()

    if not len(sys.argv) > 1:
        print("No arguments.  Exit(0).")
        exit(0)

    # Presets
    temp = args.temp or default_temp
    model = args.see or default_llm
    k = args.knoodles or default_k

    # Re-Edit GUIDE String
    if args.impress:
        impress = args.impress
    else:
        impress = GUIDE

    # Error - Cannot have both grind and wrench at the same time
    if args.grind and args.wrench:
        print("You must create vector store first before you can open it to ask question. Use -g, --grind ONLY first.")
        exit(False)

    # Error - No files to grind; missing src directory
    biz = str(args.biz) if args.biz else default_src
    if args.biz and not os.path.exists(biz):
        print("No monkey business in " + biz)
        exit(False)

    # Error - Organ does not exist; will create organ directory
    organ = str(args.organ) if args.organ else default_vdb
    if args.organ and not os.path.exists(organ):
        print(f"Vector Store {organ} does not exist. Creating directory.")
        os.mkdir(organ)

    # Merge Mode
    index = None
    if args.merge and args.organ:
        merge = str(args.merge)
        print(f"Merging {merge} into {organ}.  New vdb in {organ}_")
        nodes = []
        for path in [organ, merge]:
            # read stored index from file
            storage_context = StorageContext.from_defaults(persist_dir=path)
            index = load_index_from_storage(storage_context)

            vector_store_dict = index.storage_context.vector_store.to_dict()
            embedding_dict = vector_store_dict['embedding_dict']
            for doc_id, node in index.storage_context.docstore.docs.items():
                # necessary to avoid re-calc of embeddings
                node.embedding = embedding_dict[doc_id]
                nodes.append(node)

        merged_index = VectorStoreIndex(nodes=nodes)
        # save out to organ_

        organ = organ + "_"
        merged_index.storage_context.persist(persist_dir=organ)
        exit(True)

    if args.grind:
        print("Create vector store using PDFs in: " + biz)

        if torch.cuda.is_available():
            print("CUDA is available. PyTorch is using CUDA.")
            print(f"Number of GPUs available: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("CUDA is not available. PyTorch is using the CPU.")

        # Pre_process; convert everything to .txt to simplify ingestion
        convert_files_to_text(biz)

        # Create vector store with the .txt files
        documents = SimpleDirectoryReader(input_dir=biz,
                                          file_metadata=get_meta,
                                          recursive=True,
                                          required_exts=[".txt", ".md", ".mbox", ".epub"]
                                          ).load_data()
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=organ)
        print(f"Vector store created for {len(documents)} documents.  You can now use monkey wrench.")
        exit(True)

    if args.wrench:
        print("Loading vector store from:\t" + organ)
        storage_context = StorageContext.from_defaults(persist_dir=organ)
        index = load_index_from_storage(storage_context)

    if args.pmode:
        pmode(str(args.pmode), model)
    else:
        print(f"LLM Model:\t\t\t{model} | {temp}")
        llm = LlamaIndexOllama(model=model, temperature=temp)

        # query_engine = index.as_query_engine(llm) -- deprecated in favor of CitationQueryEngine
        # https: // docs.llamaindex.ai / en / stable / examples / query_engine / citation_query_engine /
        query_engine = CitationQueryEngine.from_args(
            index, llm,
            similarity_top_k=k,
            citation_chunk_size=1024,
            # here we can control how granular citation sources are, the default is 512
        )

        if not args.do:
            print("The monkey is interactive!  Say 'quit' to exit")

        while True:
            if args.do:
                print("\n>>> Question <<<\n" + args.do)
                prompt = impress + args.do
            else:
                user_input = input("\n>>> Question (type your question below) <<<\n")
                if user_input.lower() == 'quit':
                    exit(True)
                prompt = impress + user_input

            start_time = time.time()
            response = query_engine.query(prompt)
            elapsed_time = time.time() - start_time

            rex = textwrap.wrap(response.response, line_width)
            print("\n>>> Answer <<<")
            for i in range(len(rex)):
                print(rex[i])

            print(f"\n>>> {organ} | {model} | {temp} ({elapsed_time:.2f}s) <<<")

            for i in range(len(response.source_nodes)):
                print(f"\t | Source: {i + 1} {os.path.basename(response.source_nodes[i].node.get_metadata_str())}")
                if args.verbose:
                    print(f"\t\t | Fragment: {response.source_nodes[i].node.get_text()}")
            if args.do:
                exit(True)


if __name__ == "__main__":
        main()
