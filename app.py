import validators
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
import traceback
import yt_dlp


st.set_page_config(page_title="Langchain: Summarize Text from YouTube or any website URL")
st.title("LangChain: Summarize Text From YouTube or Website")
st.subheader("Summarize URL or YouTube Video")


with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")


generic_url = st.text_input("URL or YouTube Video URL", label_visibility="collapsed")


if groq_api_key.strip():
    llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
else:
    st.error("Please provide a valid Groq API Key")


prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


if st.button("Summarize the content from YouTube or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started.")
    elif not validators.url(generic_url):
        st.error("Please provide a valid URL.")
    else:
        try:
            with st.spinner("Summarizing the content..."):
                if "youtube.com" in generic_url:
                 
                    ydl_opts = {
                        'quiet': True,
                        'skip_download': True,
                        'format': 'bestaudio/best',
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(generic_url, download=False)
                        video_title = info.get("title", "No Title Available")
                        video_description = info.get("description", "No Description Available")
                        video_content = f"Title: {video_title}\n\nDescription: {video_description}"
                        docs = [Document(page_content=video_content)]
                else:
                    
                    from langchain_community.document_loaders import UnstructuredURLLoader
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verification=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                                          "Chrome/116.0.0.0 Safari/537.36"
                        },
                    )
                    docs = loader.load()

                
                chain = load_summarize_chain(llm, chain_type="stuff", verbose=True)
                output_summary = chain.run(docs)

                
                st.success(output_summary)

        except Exception as e:
            
            st.error(f"An error occurred: {e}")
            st.text_area("Stack Trace", traceback.format_exc(), height=300)
