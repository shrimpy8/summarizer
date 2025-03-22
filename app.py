import streamlit as st
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import tempfile
import uuid


# Read LLM API keys
load_dotenv()

st.title("Summerizer App")
st.divider()

st.markdown("## Start summerizing your documents.")

# Upload a file
uploaded_file = st.file_uploader("Upload a Text, PDF, of CSV file.",type=["txt", "pdf", "csv"], 
                                 help= "Select a file to process. Supported format: TXT, PDF, CSV")

# Initialize the model 
# Graq is free but number of token may be limited

# llm = ChatGroq(model="llama-3.3-70b-versatile")
llm = ChatOpenAI(model="gpt-4o-mini")

parser = StrOutputParser()

prompt_template = ChatPromptTemplate.from_template("Summerize the following document{document}")

# Create a chain 
chain =  prompt_template | llm | parser

if uploaded_file is not None:
    with st.spinner("Processing the uploaded file..."):
        try:
            # Create a safe temporary file with unique name
            file_extension = os.path.splitext(uploaded_file.name)[1]
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_extension}")
            
            # Log information (for debugging)
            st.info(f"Processing file: {uploaded_file.name} ({uploaded_file.type})")

            print("File: ", uploaded_file.name)
            print("File type: ", uploaded_file.type)
            print("File path: ", temp_file_path)
            #temp_file_path = uploaded_file.name
         

            # Save uplodaed file
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                # Create document loader
                if uploaded_file.type == "text/plain":
                    # Text loader

                    print("Processing TEXT file...")
                    loader = TextLoader(temp_file_path)
                elif uploaded_file.type == "text/csv":
                    # csv loader
                    print("Processing CSV file...")
                    loader = CSVLoader(temp_file_path)
                elif uploaded_file.type == "application/pdf" :
                    # pdf loader
                    print("Processing PDF file...")
                    loader = PyPDFLoader(temp_file_path)
                else:
                    st.error(f"File type '{uploaded_file.type}' is not supported!")
                    st.stop()

                # Create docment
                doc = loader.load()

                # use it for debugging 
                # print(doc)
                # result = chain.invoke({"document": doc })
                # print ("RESULT: ", result)
        
            # Text Splitter, keeping chunk size smaller
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

                # Create document chunks
                chunks = text_splitter.split_documents(doc)
                # Verify
                # print(chunks)
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    print(f"Removed temporary file: {temp_file_path}")  
  
        except Exception as e:
            # Show errors to the user instead of just printing to console
            st.error(f"An error occurred: {str(e)}")
            print(str(e))

    st.success("File uploaded ")

# Summerize

if st.button("Summerize document"):

    container = st.empty()
    
    chunk_summaries = []
    # Summerize Chunks
    with st.spinner("Summerizing chunks"):
        try:
            for i,chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}")

                # prompt
                chunk_prompt = ChatPromptTemplate.from_template(
                    "You are a highly skilled AI model tasked with summerizing text. "
                    "Please summerize the following chunk of text n a concise manner, "
                    "highlighting the most critical information. Do not omit key details: \n\n"
                    "{document}"
                )

                # Create a chunk chain
                chunk_chain =  chunk_prompt | llm | parser
                chunk_summary = chunk_chain.invoke({"document": chunk})
                chunk_summaries.append(chunk_summary)

        except Exception as e:
            print("Error summerizing chunks", {str(e)})
            st.error("Error summerizing chunks: {str(e)}")
            st.stop()

    # print("CHUNK SUMMARIES", chunk_summaries)

    # Final summary 
    with st.spinner("Creating final summary from a document..."):
        try:
            # combine all summaries
            combined_summaries = "\n".join (chunk_summaries)

            # Final summary prompt
            final_prompt = ChatPromptTemplate.from_template(
                "You are an expert summarizer tasked with creating a final summary from summarized chunks. "
                "Combine the key points from the provided summaries into a cohesive and comprehensive summary. "
                "The final summary should be concise but detailed enough to capture the main idea: \n\n {document}"

            )
            # Create final chain
            final_chain = final_prompt | llm | parser
            final_summary = final_chain.invoke({"document": combined_summaries})

            print("FINAL SUMMARY", final_summary)
            container.write(final_summary)

            # Download file 
            st.download_button(
                label="Download final summary",
                data=final_summary,
                file_name="final_summary.txt",
                mime="text/plain"
            )

        except Exception as e:
            print("Error creating final summary", {str(e)})
            st.error(f"Error creating final summary {str(e)}")



