import streamlit as st

from agent.story_agent import StoryAgent
from experiments.evaluator import StoryEvaluator


st.set_page_config(layout="wide")
st.title("Interactive Story Generation Agent")

@st.cache_resource
def load_agent():
    return StoryAgent()

agent=load_agent()
evaluator=StoryEvaluator()


# Session state
if "history" not in st.session_state:
    st.session_state.history=[]

if "single_story" not in st.session_state:
    st.session_state.single_story=None

if "experiment_results" not in st.session_state:
    st.session_state.experiment_results=None

if "model" not in st.session_state:
    st.session_state.model="Bigram"


# Sidebar
with st.sidebar:
    st.header("Generator")
    st.radio(
        "Choose model",
        [
            "Bigram",
            "Trigram",
            "RNN"
        ],
        key="model"
    )

    if st.button("Clear History"):
        st.session_state.history=[]

    st.divider()
    st.subheader("History")

    for i,item in enumerate(reversed(st.session_state.history[-8:])):
        if st.button(f"{i+1}. "+",".join(item["keywords"]), key=f"h{i}"):
            st.session_state.single_story=(item["story"])

# Tabs
tab1,tab2=st.tabs(
    [
    "Story Generator",
    "Experiment"
    ]
)


# Tab 1: Story Generator
with tab1:
    st.header("Single Model Generation")

    with st.form("generate_form"):
        prompt=st.text_input("Prompt", "cat, sun, castle")
        submitted=st.form_submit_button("Generate Story")

        if submitted:
            keywords=[
                x.strip().lower()
                for x in prompt.split(",")
                if x.strip()
            ]
            with st.spinner("Generating..."):
                story=agent.generate(keywords, st.session_state.model)

            st.session_state.single_story=story
            st.session_state.history.append({"keywords":keywords, "story":story})


    if st.session_state.single_story:
        st.subheader(st.session_state.model)
        st.write(st.session_state.single_story)


# Tab 2: Experiment
with tab2:
    st.header("Experimental Comparison")
    with st.form("experiment_form"):
        prompt2=st.text_input("Experiment prompt", "cat, sun, castle")
        runs=st.slider("Runs per model", 3, 20, 5)
        run_exp=st.form_submit_button("Run Experiment")

        if run_exp:
            keywords=[x.strip().lower() for x in prompt2.split(",") if x.strip()]

            with st.spinner("Running experiments..."):
                outputs=agent.compare_models(keywords, runs)
                analysis={}

                for model,stories in outputs.items():
                    analysis[model]=(evaluator.evaluate_runs(stories, keywords))

                st.session_state.experiment_results=(outputs, analysis)

    if st.session_state.experiment_results:
        outputs,analysis=(st.session_state.experiment_results)
        st.subheader("Metrics")

        for model,stats in analysis.items():
            model_name = model.title()  
            st.write(f"#### {model_name}")
            st.write(stats)
            st.write("Sample generations: ")

            for s in outputs[model][:3]:
                st.write("-", s)
