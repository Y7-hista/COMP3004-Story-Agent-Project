import streamlit as st
from agent.story_agent import StoryAgent

st.title("Interactive Story Agent")
st.write("Generate stories with prompts input")

@st.cache_resource
def load_agent():
    return StoryAgent()

agent = load_agent()

st.header("Input Prompts")

keywords_input = st.text_input(
    "Enter Some Prompts (Separate with commas):",
    "summer, girl"
)

generate_btn = st.button("Generate Story")

st.header("Generated Story")

if generate_btn:
    keywords = [k.strip().lower() for k in keywords_input.split(",") if k.strip()]
    story = agent.generate(keywords)
    st.write(story)