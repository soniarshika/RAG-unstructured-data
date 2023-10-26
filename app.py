import streamlit as st
from src.codellama7b import retrieval_qa, source_chain

st.title("Retrieval augmented generation using Code Llama 7b Q4_0.gguf")

# Function selection
selected_function = st.selectbox("Select a function to run:", ["retrieval_qa", "source_chain"])
query = st.text_input("Enter your query:")
creator = st.text_input("Enter creator name:")
k = st.number_input("Enter the value of k:", min_value=1, value=50)

if st.button("Run"):
    st.write(f"Running {selected_function}...")
    st.write("Please wait...")

    if selected_function == "retrieval_qa":
        result = retrieval_qa(query, creator, k)
    else:
        result = source_chain(query, creator, k)

    st.write(f"Results from {selected_function}:")

    if "answer" in result:
        st.subheader("Answer:")
        st.write(result["answer"])

    if "result" in result:
        st.subheader("Answer:")
        st.write(result["result"])

    if "sources" in result:
        st.subheader("Sources:")
        sources = result["sources"]
        for i, source in enumerate(sources, start=1):
            st.write(f"{i}. {source}")

    st.success("Finished")

st.write("Note: Select a function, enter your query, creator name, and k value, then click the 'Run' button to get results.")