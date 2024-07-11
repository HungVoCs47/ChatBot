import streamlit as st

# Load the query engine
def load_query_engine():
    from main import load
    return load()

# Use Streamlit session state to load the query engine only once
if 'query_engine' not in st.session_state:
    index = load_query_engine()
    st.session_state.query_engine = index.as_query_engine()

st.set_page_config(page_title="Your own aiChat!")

# Create a header element
st.header("Your own aiChat!")

# This sets the LLM's personality for each prompt.
# system_prompt = st.text_area(
#     label="System Prompt",
#     value="You are a helpful AI real estate agent, you should provide the information such as number of bedrooms, number of bathrooms, and the price of each real estate,",
#     key="system_prompt"
# )

# Store the conversation in the session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you today?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# Render the chat conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        
def add_to_message_history(role, content):
            message = {"role": role, "content": str(content)}
            st.session_state["messages"].append(
                message
            )  #

# Handle user input
if user_prompt := st.chat_input("Your message here", key="user_input"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    system_prompt = "You are a helpful AI real estate agent, you should provide the information such as number of bedrooms, number of bathrooms, and the price of each real estate,"

    with st.chat_message("user"):
        st.markdown(user_prompt)
        
    with st.chat_message("assistant"):
        user_prompt = system_prompt + user_prompt
        response = st.session_state["query_engine"].query(user_prompt)
        
        print(response)
        
        response_str = ""
        response_container = st.empty()
        for token in response.response:
                response_str += token
                response_container.write(response_str)
                # st.write(response.response)
        add_to_message_history("assistant", response.response)