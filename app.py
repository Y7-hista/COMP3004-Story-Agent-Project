import streamlit as st
from agent.story_agent import StoryAgent

st.title("Interactive Story Agent")
st.write("Generate stories with prompts input")

@st.cache_resource
def load_agent():
    return StoryAgent()

agent = load_agent()

st.header("Input Prompts")

def check_history(story_history):
    # story = story_history
    print("click history buttom")
    st.session_state.last_story = story_history



if "history" not in st.session_state:
    st.session_state.history = []

if "model" not in st.session_state:
    st.session_state.model = "ngram"

# Sidebar（左侧）
with st.sidebar:
    st.title("Settings")

    # ===== 模型选择 =====
    st.subheader("Model Selection")

    model = st.radio("Choose Model:", ["ngram", "random"], index=0)

    st.session_state.model = model
    # ===== 清空历史 =====
    if st.button("Clear History"):
        st.session_state.history = []

    st.divider()
    # ===== History =====
    st.subheader("History")

    for i, item in enumerate(reversed(st.session_state.history)):
        st.button(
            f"{len(st.session_state.history)-i}. {', '.join(item['keywords'])}",
            key=f"hist_{i}",
            on_click=check_history,
            args=(item['story'],)
        )

# Main Screen
st.title("Interactive Story Agent")
st.write("Generate stories with prompts input")


def generate_story():
    keywords = [k.strip().lower() for k in st.session_state.input.split(",") if k.strip()]
    if st.session_state.model == "ngram":
        story = agent.generate_ngram(keywords) 
    elif st.session_state.model == "random":
        story = "Not implemented" # agent.generate_ngram(keywords)   
    else:
        story = "Unknow Model"  
    st.session_state.last_story = story
    st.session_state.history.append({
        "keywords": keywords,
        "story": story
    })
    

st.text_input("Enter prompts", key="input")
st.button("Generate Story", on_click=generate_story)

if "last_story" in st.session_state:
    st.subheader("Generated Story")
    st.write(st.session_state.last_story)
