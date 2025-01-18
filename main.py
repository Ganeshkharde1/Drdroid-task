import os
import pandas as pd
import plotly.express as px
import torch
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings  import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator

from transformers import pipeline
from langchain import HuggingFacePipeline

def custom_wikitext_loader(file_path):
    """
    Custom loader for Wikitext files.
    Args:
        file_path: Path to the .wikitext file.

    Returns:
        The text content of the file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {e}")


def load_and_preprocess_wikitext(folder_path):
    st.write("loading dataset")
    """
    Loads and preprocesses Wikitext files from a given folder.

    Args:
        folder_path: Path to the folder containing Wikitext files.

    Returns:
        A list of Document objects.
    """
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".wikitext")]
    documents = []

    for file_path in file_paths:
        try:
            content = custom_wikitext_loader(file_path)
            documents.append(Document(page_content=content, metadata={"source": file_path}))
        except Exception as e:
            st.warning(f"Error loading file '{file_path}': {e}")

    if not documents:
        st.warning("No valid Wikitext files found in the folder.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    return chunks


def create_policy_index():
    folder_path = r"D:\My Data\study\projects\dr droid internship task\assignment_data"  # Set the path to your folder with .wikitext files
    chunks = load_and_preprocess_wikitext(folder_path)

    if not chunks:
        st.warning("No valid data to process!")
        return None

    #Loading Hugging Face model for embeddings...
    embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    #Creating ChromaDB vector store...
    vector_store_creator = VectorstoreIndexCreator(
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
        embedding=embeddings_model,
        vectorstore_cls=Chroma
    )

    #Creating index from loaded Wikitext files...
    index = vector_store_creator.from_documents(chunks)

    #Index created successfully!"
    return index


def load_llm():
    st.write("Loading LLM...")
    llm_pipeline = pipeline(
        "text-generation",
        model="microsoft/phi-2",
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=50,
        clean_up_tokenization_spaces=True
    )

    llm_model = HuggingFacePipeline(pipeline=llm_pipeline)
    st.write("LLM model loaded successfully...")
    return llm_model


def retrieve_policy_response(index, query):
    llm = load_llm()
    response = index.query(question=query, llm=llm)
    return response


def main():
    st.set_page_config(page_title="Drdroid", layout="wide")
    new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Incidents Insights -- Dashboard</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ["Dashboard", "RAG Search", "Data Viewer"])

    if options == "Dashboard":
        st.header("Data Dashboard")
        # Load your data (replace 'your_data.csv' with your actual file path)
        df = pd.read_csv('data.csv')

        # Distribution of Time to Resolve Incidents

        # Assuming 'time_to_resolve' is in a suitable time format (e.g., 'HH:MM:SS' or timedelta)
        # If not, convert it to appropriate format (e.g., timedelta)
        st.write('Distribution of Time to Resolve Incidents')
        # Create the histogram
        fig = px.histogram(df, x="time_to_resolve",
                           title="Distribution of Time to Resolve Incidents")

        # Customize the plot (optional)
        fig.update_layout(
            xaxis_title="Time to Resolve",
            yaxis_title="Number of Incidents",
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # word cloud

        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        st.write('most frequent keywords and themes related to incidents')
        # Combine all 'summary' text into a single string
        all_summaries = " ".join(df['summary'].astype(str))

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_summaries)

        # Display the word cloud in Streamlit
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        st.write('distribution of user_impact_type within different severity_level')
        # Group by 'severity_level' and count occurrences of each 'user_impact_type'
        grouped_df = df.groupby(['severity_level', 'user_impact_type']).size().reset_index(name='Count')

        # Create the stacked bar chart using Plotly Express
        fig = px.bar(
            grouped_df,
            x='severity_level',
            y='Count',
            color='user_impact_type',
            barmode='stack',
            title='Distribution of User Impact Types by Severity Level'
        )
        fig.update_traces(showlegend=False)
        # Customize the plot (optional)
        fig.update_layout(
            xaxis_title="Severity Level",
            yaxis_title="Number of Incidents",
            legend_title="User Impact Type"
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

    if options == "RAG Search":
        if 'vector_index' not in st.session_state:
            with st.spinner("📀 Wait for the RAG magic..."):
                st.session_state.vector_index = create_policy_index()


        st.header('RAG Agent')

        input_text = st.text_area("Enter your question", label_visibility="collapsed")
        go_button = st.button("🔍 Get Answer", type="primary")

        if go_button:
            with st.spinner("🔄 Processing your request... Please wait a moment."):
                if st.session_state.vector_index:
                    response_content = retrieve_policy_response(index=st.session_state.vector_index, query=input_text)
                    st.write(response_content)
                else:
                    st.warning("Please load and process the data first.")

    if options == "Data Viewer":
        df=pd.read_csv('data.csv')
        st.dataframe(df)







if __name__ == "__main__":
    main()
