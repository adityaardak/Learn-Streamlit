# app.py
# Streamlit Functions Simulation (Tutorial Playground)
# Run: streamlit run app.py

import io
import json
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
import time
import pickle
import matplotlib.pyplot as plt
import plotly.express as px

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Streamlit functions simulation",
    page_icon="🧪",
    layout="wide",
)

st.title("Streamlit funcitons simulation")  # keeping your exact title text (typo included)


# -----------------------------
# Reusable UI helpers
# -----------------------------
def show_context_box(function_name: str, uses: str, syntax: str, tips: str | None = None) -> None:
    """Header + context space (use + syntax) shown for every selected function."""
    st.subheader(function_name)

    with st.container(border=True):
        st.markdown("### ✅ Use")
        st.write(uses)

        st.markdown("### 🧾 Syntax")
        st.code(syntax, language="python")

        if tips:
            st.markdown("### 💡 Notes")
            st.write(tips)


def sample_dataframe(rows: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "id": np.arange(1, rows + 1),
            "score": np.round(rng.normal(70, 10, size=rows), 2),
            "category": rng.choice(["A", "B", "C"], size=rows),
            "date": pd.date_range("2026-01-01", periods=rows, freq="D"),
        }
    )
    return df


# -----------------------------
# Demo functions (ADD MORE HERE)
# Each demo function must:
# 1) call show_context_box(...)
# 2) provide interactive widgets below
# -----------------------------

def demo_radio_buttons():
    show_context_box(
        function_name="Radio Buttons (st.radio)",
        uses="Choose exactly one option from a small set. Great for mode selection or single-choice questions.",
        syntax=(
            "choice = st.radio(\n"
            "    label='Pick one',\n"
            "    options=['Option A', 'Option B', 'Option C'],\n"
            "    index=0,\n"
            "    horizontal=False,\n"
            "    help='Tooltip text'\n"
            ")\n"
            "st.write('You selected:', choice)"
        ),
        tips="Use `horizontal=True` for a compact UI. Use `index=None` if you want no default selection (newer Streamlit versions).",
    )

    choice = st.radio("Pick one:", ["Beginner", "Intermediate", "Advanced"], index=0, horizontal=True)
    st.write("You selected:", choice)


def demo_action_button():
    show_context_box(
        function_name="Action Button (st.button)",
        uses="Trigger an action once when clicked (run a block of logic, submit a step, start a process).",
        syntax=(
            "clicked = st.button(\n"
            "    label='Run',\n"
            "    type='primary',\n"
            "    disabled=False,\n"
            "    use_container_width=False\n"
            ")\n"
            "if clicked:\n"
            "    st.success('Button clicked!')"
        ),
        tips="Buttons re-run the script. Store results in `st.session_state` if you need persistence.",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Run action", type="primary"):
            st.session_state["last_action_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with col2:
        st.info("Tip: Click the button and notice the app reruns.")
        st.write("Last action time:", st.session_state.get("last_action_time", "—"))


def demo_text_input():
    show_context_box(
        function_name="Text Input (st.text_input)",
        uses="Take short text input like name, email, search query, etc.",
        syntax=(
            "name = st.text_input(\n"
            "    label='Your name',\n"
            "    value='',\n"
            "    max_chars=30,\n"
            "    placeholder='Type here...',\n"
            "    help='Short hint'\n"
            ")\n"
            "st.write('Hello', name)"
        ),
    )

    name = st.text_input("Your name", placeholder="Type here...")
    if name:
        st.success(f"Hello, {name} 👋")


def demo_number_input():
    show_context_box(
        function_name="Number Input (st.number_input)",
        uses="Numeric input with min/max/step. Good for hyperparameters like k, depth, learning rate, etc.",
        syntax=(
            "k = st.number_input(\n"
            "    label='K value',\n"
            "    min_value=1,\n"
            "    max_value=50,\n"
            "    value=5,\n"
            "    step=1\n"
            ")\n"
            "st.write('K =', k)"
        ),
    )

    k = st.number_input("K value", min_value=1, max_value=50, value=5, step=1)
    lr = st.number_input("Learning rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f")
    st.write({"k": int(k), "learning_rate": float(lr)})


def demo_slider():
    show_context_box(
        function_name="Slider (st.slider)",
        uses="Pick a value or range interactively. Useful for thresholds, filters, and tuning.",
        syntax=(
            "threshold = st.slider(\n"
            "    label='Threshold',\n"
            "    min_value=0.0,\n"
            "    max_value=1.0,\n"
            "    value=0.5,\n"
            "    step=0.05\n"
            ")\n"
            "st.write('Threshold:', threshold)"
        ),
    )

    threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05)
    st.progress(int(threshold * 100))
    st.write("Threshold:", threshold)


def demo_selectbox():
    show_context_box(
        function_name="Selectbox (st.selectbox)",
        uses="Dropdown single selection. Great when many options exist.",
        syntax=(
            "opt = st.selectbox(\n"
            "    label='Pick one',\n"
            "    options=['A', 'B', 'C'],\n"
            "    index=0\n"
            ")\n"
            "st.write(opt)"
        ),
    )

    opt = st.selectbox("Pick a dataset", ["Iris", "Wine", "Titanic", "Tips"], index=0)
    st.write("Selected:", opt)


def demo_multiselect():
    show_context_box(
        function_name="Multi-select (st.multiselect)",
        uses="Pick multiple options from a list (features, filters, tags).",
        syntax=(
            "features = st.multiselect(\n"
            "    label='Select features',\n"
            "    options=['f1', 'f2', 'f3'],\n"
            "    default=['f1']\n"
            ")\n"
            "st.write(features)"
        ),
    )

    features = st.multiselect("Select features", ["sepal_length", "sepal_width", "petal_length", "petal_width"], default=["sepal_length"])
    st.write("Chosen features:", features)


def demo_checkbox():
    show_context_box(
        function_name="Checkbox (st.checkbox)",
        uses="A simple true/false toggle. Great for enabling options.",
        syntax=(
            "show = st.checkbox('Show advanced options', value=False)\n"
            "if show:\n"
            "    st.write('Advanced options visible')"
        ),
    )

    show = st.checkbox("Show advanced options", value=False)
    if show:
        st.warning("Advanced options are ON")
        st.write("Example: You could show additional sliders, uploaders, etc.")


def demo_date_time_inputs():
    show_context_box(
        function_name="Date & Time Inputs (st.date_input, st.time_input)",
        uses="Collect date and time values for scheduling, filtering, logs, etc.",
        syntax=(
            "d = st.date_input('Pick a date', value=date.today())\n"
            "t = st.time_input('Pick time')\n"
            "st.write(d, t)"
        ),
    )

    d = st.date_input("Pick a date", value=date.today())
    t = st.time_input("Pick time")
    st.write("Selected:", d, t)


def demo_messages_alerts():
    show_context_box(
        function_name="Messages (st.success/info/warning/error/exception)",
        uses="Communicate status, validation, and errors to users clearly.",
        syntax=(
            "st.success('Done!')\n"
            "st.info('FYI...')\n"
            "st.warning('Be careful...')\n"
            "st.error('Something went wrong')\n"
            "try:\n"
            "    1/0\n"
            "except Exception as e:\n"
            "    st.exception(e)"
        ),
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Success"):
            st.success("Training completed successfully ✅")
    with col2:
        if st.button("Info"):
            st.info("This is an informational message ℹ️")
    with col3:
        if st.button("Warning"):
            st.warning("This is a warning ⚠️")
    with col4:
        if st.button("Error"):
            st.error("This is an error ❌")

    st.divider()
    if st.button("Show exception example"):
        try:
            _ = 1 / 0
        except Exception as e:
            st.exception(e)


def demo_columns_rows_layout():
    show_context_box(
        function_name="Columns Layout (st.columns)",
        uses="Split the page into columns for dashboards and clean layout.",
        syntax=(
            "col1, col2 = st.columns([1, 2])\n"
            "with col1:\n"
            "    st.write('Left')\n"
            "with col2:\n"
            "    st.write('Right')"
        ),
        tips="You can pass a list like [1,2,1] to control relative widths.",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.metric("Users", 1280, delta=34)
    with col2:
        df = sample_dataframe(12)
        st.dataframe(df, use_container_width=True, hide_index=True)
    with col3:
        st.write("Controls")
        st.checkbox("Enable filter")
        st.selectbox("Category", ["All", "A", "B", "C"])


def demo_tabs():
    show_context_box(
        function_name="Tabs (st.tabs)",
        uses="Organize multiple views within one screen: EDA, Training, Results, etc.",
        syntax=(
            "tab1, tab2 = st.tabs(['EDA', 'Model'])\n"
            "with tab1:\n"
            "    st.write('Charts')\n"
            "with tab2:\n"
            "    st.write('Metrics')"
        ),
    )

    tab1, tab2, tab3 = st.tabs(["EDA", "Model", "Export"])
    with tab1:
        st.write("Sample data:")
        df = sample_dataframe(25)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.line_chart(df.set_index("date")["score"])
    with tab2:
        st.write("Pretend model metrics:")
        st.metric("Accuracy", "0.89", "+0.03")
        st.metric("F1 Score", "0.86", "+0.02")
    with tab3:
        st.write("Download the sample data from the Download demo section for a full example.")


def demo_containers_expanders():
    show_context_box(
        function_name="Containers & Expanders (st.container, st.expander)",
        uses="Group UI blocks and optionally hide advanced sections.",
        syntax=(
            "with st.container(border=True):\n"
            "    st.write('Grouped content')\n"
            "with st.expander('Advanced'):\n"
            "    st.write('Hidden until opened')"
        ),
    )

    with st.container(border=True):
        st.write("This content is grouped inside a container.")
        st.text_input("Inside container input")

    with st.expander("Advanced settings"):
        st.slider("Advanced threshold", 0, 100, 50)
        st.selectbox("Advanced mode", ["Fast", "Balanced", "Accurate"])


def demo_file_uploader():
    show_context_box(
        function_name="File Upload (st.file_uploader)",
        uses="Let users upload CSV/images/audio/video, then read/process them in the app.",
        syntax=(
            "uploaded = st.file_uploader(\n"
            "    'Upload a CSV',\n"
            "    type=['csv'],\n"
            "    accept_multiple_files=False\n"
            ")\n"
            "if uploaded:\n"
            "    df = pd.read_csv(uploaded)\n"
            "    st.dataframe(df)"
        ),
        tips="For large files, add validations and show progress/spinner.",
    )

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], accept_multiple_files=False)
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success("File loaded ✅")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error("Could not read this file as CSV.")
            st.exception(e)


def demo_download_button():
    show_context_box(
        function_name="Download (st.download_button)",
        uses="Export results: CSV, JSON, text, model reports, etc.",
        syntax=(
            "csv_bytes = df.to_csv(index=False).encode('utf-8')\n"
            "st.download_button(\n"
            "    label='Download CSV',\n"
            "    data=csv_bytes,\n"
            "    file_name='data.csv',\n"
            "    mime='text/csv'\n"
            ")"
        ),
    )

    df = sample_dataframe(30)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    json_bytes = df.to_json(orient="records").encode("utf-8")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download CSV", data=csv_bytes, file_name="sample_data.csv", mime="text/csv", type="primary")
    with c2:
        st.download_button("Download JSON", data=json_bytes, file_name="sample_data.json", mime="application/json")


def demo_charts_visuals():
    show_context_box(
        function_name="Visuals (st.line_chart, st.bar_chart, st.scatter_chart)",
        uses="Quick charts without writing matplotlib/plotly code. Great for teaching and dashboards.",
        syntax=(
            "st.line_chart(df, x='date', y='score')\n"
            "st.bar_chart(df, x='category', y='score')\n"
            "st.scatter_chart(df, x='id', y='score')"
        ),
        tips="For advanced styling, use matplotlib or plotly. These quick charts are perfect for fast exploration.",
    )

    df = sample_dataframe(60)

    st.write("Pick chart type:")
    chart_type = st.radio("Chart type", ["Line", "Bar", "Scatter"], horizontal=True)

    if chart_type == "Line":
        st.line_chart(df, x="date", y="score")
    elif chart_type == "Bar":
        agg = df.groupby("category", as_index=False)["score"].mean()
        st.bar_chart(agg, x="category", y="score")
    else:
        st.scatter_chart(df, x="id", y="score")


def demo_forms():
    show_context_box(
        function_name="Forms (st.form)",
        uses="Collect multiple inputs and submit them together (like a proper form submission).",
        syntax=(
            "with st.form('my_form'):\n"
            "    name = st.text_input('Name')\n"
            "    age = st.number_input('Age', 1, 120, 18)\n"
            "    submitted = st.form_submit_button('Submit')\n"
            "if submitted:\n"
            "    st.write(name, age)"
        ),
        tips="Forms prevent re-running after every single keystroke. Everything submits together.",
    )

    with st.form("student_form"):
        name = st.text_input("Name")
        level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
        consent = st.checkbox("I confirm the data is correct")
        submitted = st.form_submit_button("Submit", type="primary")

    if submitted:
        if not consent:
            st.error("Please confirm the checkbox before submitting.")
        else:
            st.success("Form submitted ✅")
            st.json({"name": name, "level": level})


def demo_session_state():
    show_context_box(
        function_name="Session State (st.session_state)",
        uses="Store values across reruns: counters, saved inputs, app state, cached selections.",
        syntax=(
            "if 'count' not in st.session_state:\n"
            "    st.session_state['count'] = 0\n"
            "if st.button('Increment'):\n"
            "    st.session_state['count'] += 1\n"
            "st.write('Count:', st.session_state['count'])"
        ),
        tips="Session state is your best friend when teaching button clicks, multi-step flows, and persistence.",
    )

    if "count" not in st.session_state:
        st.session_state["count"] = 0

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Increment", type="primary"):
            st.session_state["count"] += 1
    with c2:
        if st.button("Decrement"):
            st.session_state["count"] -= 1
    with c3:
        if st.button("Reset"):
            st.session_state["count"] = 0

    st.write("Current count:", st.session_state["count"])


def demo_progress_spinner():
    show_context_box(
        function_name="Progress & Spinner (st.progress, st.spinner)",
        uses="Show user that something is running. Perfect for model training, downloads, processing.",
        syntax=(
            "progress = st.progress(0)\n"
            "with st.spinner('Working...'):\n"
            "    for i in range(101):\n"
            "        progress.progress(i)\n"
            "        time.sleep(0.02)\n"
            "st.success('Done!')"
        ),
    )

    if st.button("Run demo", type="primary"):
        progress = st.progress(0)
        with st.spinner("Working..."):
            for i in range(101):
                progress.progress(i)
                time.sleep(0.01)
        st.success("Done ✅")


def demo_code_markdown_json():
    show_context_box(
        function_name="Display Text/Code/Markdown/JSON (st.write, st.markdown, st.code, st.json)",
        uses="Explain concepts, show syntax, present outputs, and render structured data.",
        syntax=(
            "st.write('Hello')\n"
            "st.markdown('**Bold** and `code`')\n"
            "st.code('print(\"hi\")', language='python')\n"
            "st.json({'a': 1, 'b': 2})"
        ),
    )

    st.markdown("Here is **Markdown**, with a bullet list:\n- item 1\n- item 2")
    st.code("def add(a, b):\n    return a + b", language="python")
    st.json({"topic": "streamlit", "level": "beginner", "ok": True})


# 1️⃣ DataFrame vs Table
def demo_dataframe_vs_table():
    show_context_box(
        function_name="Data Display: st.dataframe vs st.table",
        uses="Understand difference between interactive and static table display.",
        syntax="st.dataframe(df)\nst.table(df)",
        tips="st.dataframe is interactive. st.table is static."
    )

    df = sample_dataframe(15)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### st.dataframe (Interactive)")
        st.dataframe(df, use_container_width=True)

    with col2:
        st.markdown("### st.table (Static)")
        st.table(df)


# 2️⃣ Caching Demo
@st.cache_data
def cached_load_data():
    time.sleep(2)
    return sample_dataframe(50)

def load_data_without_cache():
    time.sleep(2)
    return sample_dataframe(50)

def demo_caching():
    show_context_box(
        function_name="Caching (st.cache_data)",
        uses="Avoid recomputing expensive functions.",
        syntax="@st.cache_data\ndef load_data():\n    return pd.read_csv('file.csv')"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Load without cache"):
            start = time.time()
            load_data_without_cache()
            st.write("Time:", round(time.time() - start, 2), "seconds")

    with col2:
        if st.button("Load with cache"):
            start = time.time()
            cached_load_data()
            st.write("Time:", round(time.time() - start, 2), "seconds")


# 3️⃣ Sidebar Controls
def demo_sidebar_controls():
    show_context_box(
        function_name="Sidebar Controls",
        uses="Place filters in sidebar for dashboards.",
        syntax="value = st.sidebar.slider('Select value', 0, 100, 50)"
    )

    value = st.sidebar.slider("Sidebar Slider", 0, 100, 50)
    st.write("Selected value:", value)


# 4️⃣ Metrics Dashboard
def demo_metrics_dashboard():
    show_context_box(
        function_name="Metrics (st.metric)",
        uses="Display KPIs like Accuracy, Revenue.",
        syntax="st.metric('Accuracy', 0.92, '+0.03')"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "0.91", "+0.02")
    col2.metric("Loss", "0.12", "-0.01")
    col3.metric("Users", "1250", "+120")


# 5️⃣ Matplotlib Integration
def demo_matplotlib():
    show_context_box(
        function_name="Matplotlib (st.pyplot)",
        uses="Display custom matplotlib figures.",
        syntax="fig, ax = plt.subplots()\nax.plot(x,y)\nst.pyplot(fig)"
    )

    df = sample_dataframe(30)
    fig, ax = plt.subplots()
    ax.plot(df["id"], df["score"])
    ax.set_title("Matplotlib Plot")
    st.pyplot(fig)


# 6️⃣ Plotly Integration
def demo_plotly():
    show_context_box(
        function_name="Plotly (st.plotly_chart)",
        uses="Interactive visualizations.",
        syntax="fig = px.scatter(df, x='id', y='score')\nst.plotly_chart(fig)"
    )

    df = sample_dataframe(50)
    fig = px.scatter(df, x="id", y="score", color="category")
    st.plotly_chart(fig)


# 7️⃣ Stop Execution
def demo_stop_execution():
    show_context_box(
        function_name="Stop Execution (st.stop)",
        uses="Stop script if condition fails.",
        syntax="if not uploaded:\n    st.stop()"
    )

    uploaded = st.file_uploader("Upload CSV")

    if not uploaded:
        st.error("Upload a file first.")
        st.stop()

    df = pd.read_csv(uploaded)
    st.success("File Loaded")
    st.dataframe(df)


# 8️⃣ Model Download (PKL)
def demo_model_download():
    show_context_box(
        function_name="Download Model (PKL)",
        uses="Allow users to download trained ML models.",
        syntax="pickle.dumps(model)\nst.download_button(...)"
    )

    dummy_model = {"model": "Logistic Regression", "accuracy": 0.91}
    model_bytes = pickle.dumps(dummy_model)

    st.download_button(
        "Download Dummy Model",
        data=model_bytes,
        file_name="model.pkl",
        mime="application/octet-stream"
    )


# 9️⃣ Conditional Rendering
def demo_conditional_rendering():
    show_context_box(
        function_name="Conditional Rendering",
        uses="Display content based on user input.",
        syntax="if st.checkbox('Show chart'):\n    st.line_chart(data)"
    )

    df = sample_dataframe(40)
    if st.checkbox("Show Chart"):
        st.line_chart(df["score"])


# 🔟 Advanced Session State
def demo_session_state_advanced():
    show_context_box(
        function_name="Advanced Session State",
        uses="Store workflow state between reruns.",
        syntax="if 'trained' not in st.session_state:\n    st.session_state['trained'] = False"
    )

    if "trained" not in st.session_state:
        st.session_state["trained"] = False

    if st.button("Train Model"):
        st.session_state["trained"] = True

    if st.session_state["trained"]:
        st.success("Model trained and stored in session.")
    else:
        st.info("Model not trained yet.")

# -----------------------------
# Registry of features (Sidebar dropdown)
# Add new items here: "Menu Label": demo_function
# -----------------------------
FEATURES = {
    "Welcome": None,
    "Radio Buttons": demo_radio_buttons,
    "Action Button": demo_action_button,
    "Text Input": demo_text_input,
    "Number Input": demo_number_input,
    "Slider": demo_slider,
    "Selectbox": demo_selectbox,
    "Multi-select": demo_multiselect,
    "Checkbox": demo_checkbox,
    "Date & Time Inputs": demo_date_time_inputs,
    "Messages (Success/Info/Warning/Error)": demo_messages_alerts,
    "Columns Layout": demo_columns_rows_layout,
    "Tabs": demo_tabs,
    "Containers & Expanders": demo_containers_expanders,
    "File Upload": demo_file_uploader,
    "Download Button": demo_download_button,
    "Charts / Visuals": demo_charts_visuals,
    "Forms": demo_forms,
    "Session State": demo_session_state,
    "Progress & Spinner": demo_progress_spinner,
    "Text/Code/Markdown/JSON": demo_code_markdown_json,
     "DataFrame vs Table": demo_dataframe_vs_table,
    "Caching": demo_caching,
    "Sidebar Controls": demo_sidebar_controls,
    "Metrics Dashboard": demo_metrics_dashboard,
    "Matplotlib": demo_matplotlib,
    "Plotly": demo_plotly,
    "Stop Execution": demo_stop_execution,
    "Model Download (PKL)": demo_model_download,
    "Conditional Rendering": demo_conditional_rendering,
    "Advanced Session State": demo_session_state_advanced,
}


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("🧭 Function Selector")
selected_label = st.sidebar.selectbox("Select a Streamlit feature", list(FEATURES.keys()), index=0)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Add new demos by creating a `demo_...()` function and registering it in the `FEATURES` dictionary."
)


# -----------------------------
# Main display area
# -----------------------------
if selected_label == "Welcome":
    st.subheader("Welcome 👋")
    st.write(
        "Use the left panel to choose a Streamlit function.\n\n"
        "For each selection you will see:\n"
        "1) Function name (header)\n"
        "2) A context box with **use** and **syntax**\n"
        "3) An interactive playground to test it"
    )
    st.info("Tip: This app is intentionally modular so you can add more demo functions easily.")
else:
    # Run the selected demo function
    demo_fn = FEATURES[selected_label]
    if demo_fn:
        demo_fn()
