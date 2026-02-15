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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
 mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor




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

# ----------------------------
# 1) st.write vs st.text vs st.caption vs st.markdown
# ----------------------------
def demo_text_display_variants():
    show_context_box(
        function_name="Text Display Variants (st.write, st.text, st.caption, st.markdown)",
        uses="Teach students how to present explanations, notes, and formatting in a DS app.",
        syntax=(
            "st.write('Normal text or objects')\n"
            "st.text('Monospace plain text')\n"
            "st.caption('Small helper text')\n"
            "st.markdown('**Markdown** with `code`')"
        ),
        tips="Use st.caption for small hints; st.markdown for rich formatting; st.text for raw logs."
    )
    st.write("st.write can render text, dataframes, dicts, and more.")
    st.text("st.text shows plain monospace text (good for logs).")
    st.caption("st.caption is subtle helper text.")
    st.markdown("st.markdown supports **bold**, *italic*, `inline code`, lists, and headings.")


# ----------------------------
# 2) st.json + pretty dict outputs (for configs, metrics)
# ----------------------------
def demo_json_and_config_view():
    show_context_box(
        function_name="JSON Display (st.json) for Config & Metrics",
        uses="Show training configs, model params, API responses, evaluation metrics as structured JSON.",
        syntax="st.json({'model': 'LR', 'acc': 0.91, 'params': {'C': 1.0}})"
    )
    config = {
        "dataset": "Iris",
        "model": "LogisticRegression",
        "metrics": {"accuracy": 0.91, "f1_macro": 0.88},
        "params": {"C": 1.0, "max_iter": 200},
    }
    st.json(config)


# ----------------------------
# 3) st.empty placeholder (dynamic UI updates)
# ----------------------------
def demo_empty_placeholder():
    show_context_box(
        function_name="Dynamic Placeholders (st.empty)",
        uses="Update the same area repeatedly: live logs, live metrics, progress updates.",
        syntax=(
            "slot = st.empty()\n"
            "for i in range(5):\n"
            "    slot.info(f'Step {i}')\n"
            "    time.sleep(1)"
        ),
        tips="Great for simulated training loops and showing live metric updates."
    )

    slot = st.empty()
    if st.button("Run placeholder demo", type="primary"):
        for i in range(1, 6):
            slot.info(f"Processing step {i}/5...")
            time.sleep(0.6)
        slot.success("Done ✅")


# ----------------------------
# 4) st.toast notifications (quick feedback)
# ----------------------------
def demo_toast_notifications():
    show_context_box(
        function_name="Toast Notifications (st.toast)",
        uses="Give lightweight feedback without cluttering the page (great for DS apps).",
        syntax="st.toast('Model trained successfully!')",
        tips="Works well when user clicks buttons like Train, Export, Save settings."
    )
    if st.button("Show toast", type="primary"):
        st.toast("✅ Action completed!")
        st.toast("📌 Tip: Use toast for quick feedback.")


# ----------------------------
# 5) st.status (pipeline steps UI)
# ----------------------------
def demo_status_pipeline():
    show_context_box(
        function_name="Pipeline Status (st.status)",
        uses="Show multi-step DS pipelines (load → clean → train → evaluate → export).",
        syntax=(
            "with st.status('Running...', expanded=True) as s:\n"
            "    st.write('Step 1')\n"
            "    s.update(label='Done', state='complete')"
        ),
        tips="Perfect to teach students the real ML workflow, step-by-step."
    )

    if st.button("Run pipeline status demo", type="primary"):
        with st.status("Running DS pipeline...", expanded=True) as s:
            st.write("1) Loading data...")
            time.sleep(0.7)
            st.write("2) Cleaning data...")
            time.sleep(0.7)
            st.write("3) Training model...")
            time.sleep(0.7)
            st.write("4) Evaluating metrics...")
            time.sleep(0.7)
            st.write("5) Exporting artifacts...")
            time.sleep(0.7)
            s.update(label="Pipeline complete ✅", state="complete")


# ----------------------------
# 6) st.form + validation + st.stop patterns (hyperparameter submission)
# ----------------------------
def demo_hyperparam_form_validation():
    show_context_box(
        function_name="Hyperparameter Form + Validation (st.form, st.stop)",
        uses="Teach safe ML app patterns: collect params, validate, then train.",
        syntax=(
            "with st.form('hp'):\n"
            "    C = st.number_input('C', 0.01, 10.0, 1.0)\n"
            "    submit = st.form_submit_button('Train')\n"
            "if submit:\n"
            "    if C <= 0:\n"
            "        st.error('Invalid C'); st.stop()"
        )
    )

    with st.form("hp_form"):
        C = st.number_input("LogReg C (inverse regularization)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        max_iter = st.number_input("max_iter", min_value=50, max_value=2000, value=200, step=50)
        submit = st.form_submit_button("Validate & Save", type="primary")

    if submit:
        if C <= 0:
            st.error("C must be > 0")
            st.stop()
        st.success("Hyperparameters validated ✅")
        st.session_state["saved_hp"] = {"C": float(C), "max_iter": int(max_iter)}
        st.json(st.session_state["saved_hp"])


# ----------------------------
# 7) DS Pipeline: Load dataset (Iris/Wine) + Train + Evaluate
# ----------------------------
def _get_dataset(name: str):
    if name == "Iris":
        data = load_iris(as_frame=True)
    else:
        data = load_wine(as_frame=True)
    df = data.frame.copy()
    X = df.drop(columns=[data.target_names is None and "target" or "target"]) if "target" in df.columns else df.drop(columns=[df.columns[-1]])
    # In sklearn as_frame, the target column is named 'target'
    X = df.drop(columns=["target"])
    y = df["target"]
    return df, X, y

def demo_train_evaluate_classification():
    show_context_box(
        function_name="Mini ML Trainer (sklearn) + Metrics",
        uses="Let students practice end-to-end: choose dataset, model, hyperparameters, train, and see report.",
        syntax=(
            "X_train, X_test, y_train, y_test = train_test_split(...)\n"
            "pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])\n"
            "pipe.fit(X_train, y_train)\n"
            "pred = pipe.predict(X_test)\n"
            "st.write(accuracy_score(y_test, pred))\n"
            "st.text(classification_report(y_test, pred))"
        ),
        tips="This is a real DS app pattern: sidebar config → train button → metrics + confusion matrix."
    )

    left, right = st.columns([1, 2])

    with left:
        dataset_name = st.selectbox("Dataset", ["Iris", "Wine"])
        model_name = st.selectbox("Model", ["Logistic Regression", "Decision Tree"])

        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", 0, 9999, 42, 1)

        use_scaler = st.checkbox("Use StandardScaler (recommended for LR)", value=True)

        if model_name == "Logistic Regression":
            C = st.number_input("C", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
            max_iter = st.number_input("max_iter", min_value=50, max_value=2000, value=200, step=50)
        else:
            max_depth = st.number_input("max_depth", min_value=1, max_value=30, value=4, step=1)
            min_samples_split = st.number_input("min_samples_split", min_value=2, max_value=50, value=2, step=1)

        train_btn = st.button("Train & Evaluate", type="primary")

    with right:
        df, X, y = _get_dataset(dataset_name)
        st.markdown("### Dataset preview")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        if train_btn:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
            )

            if model_name == "Logistic Regression":
                model = LogisticRegression(C=float(C), max_iter=int(max_iter))
                if use_scaler:
                    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
                else:
                    pipe = Pipeline([("model", model)])
            else:
                model = DecisionTreeClassifier(
                    max_depth=int(max_depth),
                    min_samples_split=int(min_samples_split),
                    random_state=int(random_state),
                )
                pipe = Pipeline([("model", model)])

            with st.spinner("Training model..."):
                time.sleep(0.4)
                pipe.fit(X_train, y_train)

            pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, pred)
            st.success("Training complete ✅")

            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("Train rows", f"{len(X_train)}")
            c3.metric("Test rows", f"{len(X_test)}")

            st.markdown("### Classification report")
            st.text(classification_report(y_test, pred))

            st.markdown("### Confusion matrix")
            cm = confusion_matrix(y_test, pred)
            st.dataframe(pd.DataFrame(cm), use_container_width=True)

            st.session_state["last_model"] = pipe
            st.session_state["last_metrics"] = {"accuracy": float(acc), "dataset": dataset_name, "model": model_name}


# ----------------------------
# 8) Download last trained model + metrics (PKL/JSON)
# ----------------------------
def demo_export_artifacts():
    show_context_box(
        function_name="Export Artifacts (Download Model + Metrics)",
        uses="Teach how DS apps export models and evaluation results for submissions or deployment.",
        syntax=(
            "model_bytes = pickle.dumps(model)\n"
            "st.download_button('Download model', model_bytes, 'model.pkl')\n"
            "st.download_button('Download metrics', json_bytes, 'metrics.json')"
        ),
        tips="This works best after running the Mini ML Trainer demo once."
    )

    model = st.session_state.get("last_model")
    metrics = st.session_state.get("last_metrics")

    if not model or not metrics:
        st.warning("No trained model found in session. First run: 'Mini ML Trainer (sklearn) + Metrics'.")
        st.stop()

    model_bytes = pickle.dumps(model)
    metrics_bytes = json.dumps(metrics, indent=2).encode("utf-8")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download model.pkl",
            data=model_bytes,
            file_name="model.pkl",
            mime="application/octet-stream",
            type="primary",
        )
    with c2:
        st.download_button(
            "Download metrics.json",
            data=metrics_bytes,
            file_name="metrics.json",
            mime="application/json",
        )

    st.json(metrics)


# ----------------------------
# 9) EDA: describe + missing + correlation heatmap (matplotlib)
# ----------------------------
def demo_eda_quick_lab():
    show_context_box(
        function_name="Quick EDA Lab (Describe, Missing, Correlation)",
        uses="Teach students the first 10 minutes of any DS project inside Streamlit.",
        syntax=(
            "st.dataframe(df.head())\n"
            "st.write(df.describe())\n"
            "st.write(df.isna().sum())\n"
            "st.pyplot(fig)"
        ),
        tips="Use this pattern with uploaded CSVs too (connect with file_uploader)."
    )

    df = sample_dataframe(80).copy()
    # Add a few missing values for teaching
    df.loc[df.sample(5, random_state=7).index, "score"] = np.nan

    st.markdown("### Preview")
    st.dataframe(df.head(12), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Summary (describe)")
        st.dataframe(df.describe(include="all"), use_container_width=True)

    with c2:
        st.markdown("### Missing values")
        st.dataframe(df.isna().sum().to_frame("missing_count"), use_container_width=True)

    st.markdown("### Correlation heatmap (numeric)")
    num_df = df.select_dtypes(include=[np.number]).fillna(df.select_dtypes(include=[np.number]).mean())
    corr = num_df.corr()

    fig, ax = plt.subplots()
    im = ax.imshow(corr.values)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)


# ----------------------------
# 10) Data Filtering Lab (query, slider, categorical filter)
# ----------------------------
def demo_data_filtering_lab():
    show_context_box(
        function_name="Data Filtering Lab (Dashboard Filters)",
        uses="Teach students real dashboard filtering: numeric sliders + category selectors + search.",
        syntax=(
            "q = st.text_input('Search')\n"
            "min_score, max_score = st.slider('Score range', ...)\n"
            "cats = st.multiselect('Category', ...)\n"
            "filtered = df[...]\n"
            "st.dataframe(filtered)"
        )
    )

    df = sample_dataframe(120)

    q = st.text_input("Search by category (A/B/C) or leave blank", placeholder="Example: A")
    min_score, max_score = st.slider("Score range", 0.0, 100.0, (40.0, 90.0), 1.0)
    cats = st.multiselect("Category filter", sorted(df["category"].unique()), default=sorted(df["category"].unique()))

    filtered = df[df["category"].isin(cats)].copy()
    filtered = filtered[(filtered["score"] >= min_score) & (filtered["score"] <= max_score)]
    if q.strip():
        filtered = filtered[filtered["category"].str.contains(q.strip(), case=False, na=False)]

    st.metric("Rows after filter", len(filtered))
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.line_chart(filtered.set_index("date")["score"])


# ----------------------------
# 11) Plotly EDA Lab (interactive histogram + scatter)
# ----------------------------
def demo_plotly_eda_lab():
    show_context_box(
        function_name="Interactive Plotly EDA Lab",
        uses="Teach hover/zoom charts for EDA: histogram and scatter with category color.",
        syntax=(
            "fig = px.histogram(df, x='score', color='category')\n"
            "st.plotly_chart(fig)\n"
            "fig2 = px.scatter(df, x='id', y='score', color='category')\n"
            "st.plotly_chart(fig2)"
        )
    )

    df = sample_dataframe(200)
    fig = px.histogram(df, x="score", color="category", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(df, x="id", y="score", color="category")
    st.plotly_chart(fig2, use_container_width=True)


# ----------------------------
# 12) Button actions with st.rerun pattern (safe)
# ----------------------------
def demo_rerun_pattern():
    show_context_box(
        function_name="Rerun Pattern (Action → Update State → Rerun)",
        uses="Teach multi-step DS flows where you update session_state and immediately refresh UI.",
        syntax=(
            "if st.button('Reset filters'):\n"
            "    st.session_state['x'] = 0\n"
            "    st.rerun()"
        ),
        tips="Use this to reset dashboards or clear previous model results."
    )

    if "rerun_count" not in st.session_state:
        st.session_state["rerun_count"] = 0

    if st.button("Increment + rerun", type="primary"):
        st.session_state["rerun_count"] += 1
        st.rerun()

    st.write("Rerun counter:", st.session_state["rerun_count"])

def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _get_iris():
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    X = df.drop(columns=["target"])
    y = df["target"]
    return df, X, y

def _get_wine():
    data = load_wine(as_frame=True)
    df = data.frame.copy()
    X = df.drop(columns=["target"])
    y = df["target"]
    return df, X, y

def _get_diabetes_reg():
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    X = df.drop(columns=["target"])
    y = df["target"]
    return df, X, y


# ================================
# 1) Regression Trainer + Metrics (MAE/RMSE/R2)
# ================================
def demo_regression_trainer():
    show_context_box(
        function_name="Regression Trainer (Linear/Ridge/Lasso) + MAE/RMSE/R²",
        uses="Teach DS students regression workflow, evaluation metrics, and regularization impact.",
        syntax=(
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
            "pipe = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))])\n"
            "pipe.fit(X_train, y_train)\n"
            "pred = pipe.predict(X_test)\n"
            "mae = mean_absolute_error(y_test, pred)\n"
            "rmse = sqrt(mean_squared_error(y_test, pred))\n"
            "r2 = r2_score(y_test, pred)"
        ),
        tips="Use Ridge/Lasso to show how regularization changes model behavior."
    )

    df, X, y = _get_diabetes_reg()
    st.markdown("### Dataset: Diabetes (built-in sklearn regression dataset)")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        model_name = st.selectbox("Model", ["Linear Regression", "Ridge", "Lasso"])
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", 0, 9999, 42, 1)
        use_scaler = st.checkbox("Use StandardScaler", value=True)

        alpha = None
        if model_name in ["Ridge", "Lasso"]:
            alpha = st.slider("alpha (regularization strength)", 0.0, 5.0, 1.0, 0.1)

        train_btn = st.button("Train Regression Model", type="primary")

    with col2:
        if train_btn:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_size), random_state=int(random_state)
            )

            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Ridge":
                model = Ridge(alpha=float(alpha))
            else:
                model = Lasso(alpha=float(alpha), max_iter=5000)

            steps = []
            if use_scaler:
                steps.append(("scaler", StandardScaler()))
            steps.append(("model", model))
            pipe = Pipeline(steps)

            with st.spinner("Training regression model..."):
                time.sleep(0.4)
                pipe.fit(X_train, y_train)

            pred = pipe.predict(X_test)

            mae = float(mean_absolute_error(y_test, pred))
            rmse = _rmse(y_test, pred)
            r2 = float(r2_score(y_test, pred))

            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{mae:.3f}")
            c2.metric("RMSE", f"{rmse:.3f}")
            c3.metric("R²", f"{r2:.3f}")

            st.session_state["last_reg_model"] = pipe
            st.session_state["last_reg_metrics"] = {"MAE": mae, "RMSE": rmse, "R2": r2, "model": model_name}

            # Actual vs Pred plot
            fig, ax = plt.subplots()
            ax.scatter(y_test, pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)


# ================================
# 2) Cross-Validation Lab (KFold vs StratifiedKFold)
# ================================
def demo_cross_validation_lab():
    show_context_box(
        function_name="Cross-Validation Lab (KFold vs StratifiedKFold)",
        uses="Teach model evaluation beyond a single split and why stratification matters in classification.",
        syntax=(
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
            "scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')\n"
            "st.write(scores.mean())"
        ),
        tips="Use KFold for regression; StratifiedKFold for classification to keep class balance per fold."
    )

    dataset = st.selectbox("Dataset", ["Iris", "Wine"])
    if dataset == "Iris":
        _, X, y = _get_iris()
    else:
        _, X, y = _get_wine()

    model = LogisticRegression(max_iter=200)

    col1, col2 = st.columns(2)
    with col1:
        n_splits = st.slider("n_splits", 2, 10, 5, 1)
        shuffle = st.checkbox("shuffle", value=True)
        rs = st.number_input("random_state", 0, 9999, 42, 1)

        run_kfold = st.button("Run KFold", type="primary")

    with col2:
        run_strat = st.button("Run StratifiedKFold", type="primary")

    if run_kfold:
        cv = KFold(n_splits=int(n_splits), shuffle=bool(shuffle), random_state=int(rs) if shuffle else None)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        st.success("KFold completed ✅")
        st.write("Scores:", np.round(scores, 4))
        st.metric("Mean accuracy", f"{scores.mean():.4f}")
        st.metric("Std", f"{scores.std():.4f}")

    if run_strat:
        cv = StratifiedKFold(n_splits=int(n_splits), shuffle=bool(shuffle), random_state=int(rs) if shuffle else None)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        st.success("StratifiedKFold completed ✅")
        st.write("Scores:", np.round(scores, 4))
        st.metric("Mean accuracy", f"{scores.mean():.4f}")
        st.metric("Std", f"{scores.std():.4f}")


# ================================
# 3) Data Leakage Demo (Wrong vs Right scaling)
# ================================
def demo_data_leakage_scaling():
    show_context_box(
        function_name="Data Leakage Demo (Scaling Wrong vs Right)",
        uses="Teach a common DS mistake: fitting scaler on full data before split causes leakage.",
        syntax=(
            "# WRONG:\n"
            "scaler.fit(X)  # uses test info\n"
            "X_scaled = scaler.transform(X)\n"
            "...\n\n"
            "# RIGHT:\n"
            "pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])\n"
            "pipe.fit(X_train, y_train)"
        ),
        tips="Always preprocess inside a Pipeline so CV/train-test splitting stays clean."
    )

    _, X, y = _get_iris()
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    rs = st.number_input("random_state", 0, 9999, 42, 1)

    if st.button("Run Leakage Demo", type="primary"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=int(rs), stratify=y)

        # WRONG approach
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # leakage: uses all data
        X_tr_w, X_te_w, y_tr_w, y_te_w = train_test_split(X_scaled, y, test_size=float(test_size), random_state=int(rs), stratify=y)
        wrong_model = LogisticRegression(max_iter=200).fit(X_tr_w, y_tr_w)
        wrong_acc = accuracy_score(y_te_w, wrong_model.predict(X_te_w))

        # RIGHT approach
        right_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=200))])
        right_pipe.fit(X_train, y_train)
        right_acc = accuracy_score(y_test, right_pipe.predict(X_test))

        c1, c2 = st.columns(2)
        c1.metric("WRONG (leakage) accuracy", f"{wrong_acc:.4f}")
        c2.metric("RIGHT (pipeline) accuracy", f"{right_acc:.4f}")
        st.info("Even if scores look similar sometimes, leakage is still invalid and can inflate results in real projects.")


# ================================
# 4) Feature Scaling Visual Demo (before/after)
# ================================
def demo_scaling_visual():
    show_context_box(
        function_name="Feature Scaling Visual (Before vs After)",
        uses="Show why scaling matters: compare distributions before and after StandardScaler.",
        syntax=(
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X)\n"
            "st.dataframe(pd.DataFrame(X_scaled, columns=X.columns))"
        ),
        tips="Great when teaching LR, SVM, KNN."
    )

    _, X, _ = _get_wine()
    feature = st.selectbox("Select feature", list(X.columns), index=0)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Before Scaling")
        fig1 = px.histogram(X, x=feature, nbins=25)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("### After Scaling")
        fig2 = px.histogram(X_scaled, x=feature, nbins=25)
        st.plotly_chart(fig2, use_container_width=True)


# ================================
# 5) Outlier Detection Lab (IQR method) + visualization
# ================================
def demo_outlier_detection_iqr():
    show_context_box(
        function_name="Outlier Detection (IQR Method)",
        uses="Teach simple outlier detection and how removing outliers changes distribution.",
        syntax=(
            "Q1 = df[col].quantile(0.25)\n"
            "Q3 = df[col].quantile(0.75)\n"
            "IQR = Q3 - Q1\n"
            "lower = Q1 - 1.5*IQR\n"
            "upper = Q3 + 1.5*IQR\n"
            "clean = df[(df[col]>=lower) & (df[col]<=upper)]"
        )
    )

    df = sample_dataframe(250)
    col = st.selectbox("Column", ["score"], index=0)

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    clean = df[(df[col] >= lower) & (df[col] <= upper)].copy()
    outliers = df[(df[col] < lower) | (df[col] > upper)].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total rows", len(df))
    c2.metric("Outliers", len(outliers))
    c3.metric("After removal", len(clean))

    fig = px.box(df, y=col, points="all", title="Box plot (outliers visible)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Thresholds")
    st.write({"lower": float(lower), "upper": float(upper)})


# ================================
# 6) Missing Value Handling Lab (mean/median/mode)
# ================================
def demo_missing_value_imputation():
    show_context_box(
        function_name="Missing Value Handling (Mean/Median/Mode)",
        uses="Teach core preprocessing: identify missing values and fill them using common strategies.",
        syntax=(
            "df[col] = df[col].fillna(df[col].mean())\n"
            "df[col] = df[col].fillna(df[col].median())\n"
            "df[col] = df[col].fillna(df[col].mode()[0])"
        )
    )

    df = sample_dataframe(120).copy()
    # Inject missing values
    df.loc[df.sample(12, random_state=7).index, "score"] = np.nan

    strategy = st.selectbox("Strategy", ["Mean", "Median", "Mode"])
    before_missing = int(df["score"].isna().sum())

    if strategy == "Mean":
        df["score"] = df["score"].fillna(df["score"].mean())
    elif strategy == "Median":
        df["score"] = df["score"].fillna(df["score"].median())
    else:
        df["score"] = df["score"].fillna(df["score"].mode()[0])

    after_missing = int(df["score"].isna().sum())

    c1, c2 = st.columns(2)
    c1.metric("Missing before", before_missing)
    c2.metric("Missing after", after_missing)

    st.dataframe(df.head(20), use_container_width=True, hide_index=True)


# ================================
# 7) Encoding Lab (One-Hot for category)
# ================================
def demo_one_hot_encoding():
    show_context_box(
        function_name="Categorical Encoding (One-Hot Encoding)",
        uses="Teach conversion of categorical columns into numeric features for ML models.",
        syntax="encoded = pd.get_dummies(df, columns=['category'], drop_first=True)"
    )

    df = sample_dataframe(30)[["id", "score", "category"]].copy()
    st.markdown("### Original data")
    st.dataframe(df, use_container_width=True, hide_index=True)

    drop_first = st.checkbox("drop_first=True", value=True)
    encoded = pd.get_dummies(df, columns=["category"], drop_first=drop_first)

    st.markdown("### After One-Hot Encoding")
    st.dataframe(encoded, use_container_width=True, hide_index=True)


# ================================
# 8) Train/Test Split Visual (class distribution)
# ================================
def demo_split_distribution_visual():
    show_context_box(
        function_name="Train/Test Split Visual (Class Distribution)",
        uses="Teach why stratify is important: class balance in train and test splits.",
        syntax=(
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "   X, y, test_size=0.2, stratify=y, random_state=42\n"
            ")"
        )
    )

    dataset = st.selectbox("Dataset", ["Iris", "Wine"])
    if dataset == "Iris":
        _, X, y = _get_iris()
    else:
        _, X, y = _get_wine()

    test_size = st.slider("test_size", 0.1, 0.5, 0.2, 0.05)
    rs = st.number_input("random_state", 0, 9999, 42, 1)
    use_stratify = st.checkbox("Use stratify=y", value=True)

    if st.button("Split & Visualize", type="primary"):
        strat = y if use_stratify else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=int(rs), stratify=strat)

        train_counts = y_train.value_counts().sort_index()
        test_counts = y_test.value_counts().sort_index()

        df_counts = pd.DataFrame({"train": train_counts, "test": test_counts}).fillna(0).astype(int)
        st.dataframe(df_counts, use_container_width=True)

        fig = px.bar(df_counts.reset_index().melt(id_vars="index", var_name="split", value_name="count"),
                     x="index", y="count", color="split", barmode="group", title="Class Distribution")
        fig.update_xaxes(title="Class")
        st.plotly_chart(fig, use_container_width=True)


# ================================
# 9) Decision Boundary (2 features) - Logistic Regression on Iris
# ================================
def demo_decision_boundary_2d():
    show_context_box(
        function_name="Decision Boundary (2D) - Classification Visual",
        uses="Teach how model learns separation when using only 2 features (classic DS visualization).",
        syntax=(
            "Use 2 columns from X\n"
            "Train model\n"
            "Create meshgrid\n"
            "Predict on grid\n"
            "Plot decision regions"
        ),
        tips="A great classroom demo for how adding features changes model behavior."
    )

    df, X, y = _get_iris()
    features = st.multiselect("Pick exactly 2 features", list(X.columns), default=list(X.columns)[:2])

    if len(features) != 2:
        st.warning("Please select exactly 2 features.")
        st.stop()

    test_size = st.slider("test_size", 0.1, 0.5, 0.2, 0.05)
    rs = st.number_input("random_state", 0, 9999, 42, 1)

    if st.button("Plot Decision Boundary", type="primary"):
        X2 = X[features].copy()
        X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=float(test_size), random_state=int(rs), stratify=y)

        pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=300))])
        pipe.fit(X_train, y_train)

        # Meshgrid
        x_min, x_max = X2[features[0]].min() - 0.5, X2[features[0]].max() + 0.5
        y_min, y_max = X2[features[1]].min() - 0.5, X2[features[1]].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))

        grid = pd.DataFrame({features[0]: xx.ravel(), features[1]: yy.ravel()})
        zz = pipe.predict(grid).reshape(xx.shape)

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, zz, alpha=0.25)
        scatter = ax.scatter(X2[features[0]], X2[features[1]], c=y, s=30)
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_title("Decision Regions (LogReg) with 2 Features")
        st.pyplot(fig)

        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        st.metric("Test Accuracy", f"{acc:.3f}")



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
    "Text Display Variants": demo_text_display_variants,
    "JSON / Config Viewer": demo_json_and_config_view,
    "Dynamic Placeholder (st.empty)": demo_empty_placeholder,
    "Toast Notifications": demo_toast_notifications,
    "Pipeline Status (st.status)": demo_status_pipeline,
    "Hyperparameter Form + Validation": demo_hyperparam_form_validation,
    "Plotly EDA Lab": demo_plotly_eda_lab,
    "Rerun Pattern": demo_rerun_pattern,
    
}

DS_DEMOS = {
    "EDA Dashboard Demo (filters + charts)": demo_data_filtering_lab,
    "Mini ML Trainer (train + metrics)": demo_train_evaluate_classification,
    "Quick EDA Lab (missing + corr)": demo_eda_quick_lab,
    "Export Artifacts Demo (download model + json)": demo_export_artifacts,
    "Outlier Detection Demo (IQR)": demo_outlier_detection_iqr,
    "Regression Trainer (MAE/RMSE/R²)": demo_regression_trainer,
    "Cross-Validation Lab": demo_cross_validation_lab,
    "Data Leakage Demo (Scaling)": demo_data_leakage_scaling,
    "Scaling Visual (Before/After)": demo_scaling_visual,
    "Missing Value Imputation": demo_missing_value_imputation,
    "One-Hot Encoding": demo_one_hot_encoding,
    "Split Distribution Visual": demo_split_distribution_visual,
    "Decision Boundary (2D)": demo_decision_boundary_2d,
}

# -----------------------------
# Sidebar
# -----------------------------
# ================================
# ✅ SIDEBAR: 2-level selector
# 1) Streamlit Function Selector
# 2) Data Science Demo Selector (contextual use)
# ================================

st.sidebar.header("🧭 Function Selector")
selected_function = st.sidebar.selectbox(
    "Select a Streamlit function",
    list(FEATURES.keys()),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Data Science Demos")

# These demos are NOT “streamlit functions”, they are practical DS mini-scenarios
# that show how the selected streamlit function becomes useful in real apps.
selected_demo = st.sidebar.selectbox(
    "Run a DS demo (practical use-case)",
    ["None"] + list(DS_DEMOS.keys()),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Learn the function first, then run a DS demo to see why it matters.")


# ================================
# ✅ MAIN: routing
# Priority: if DS demo selected → run it
# else → run selected function demo
# ================================
if selected_demo != "None":
    DS_DEMOS[selected_demo]()     # DS use-case demo
else:
    if selected_function == "Welcome":
        st.subheader("Welcome 👋")
        st.write(
            "Step 1: Select a Streamlit function from the left.\n"
            "Step 2: (Optional) Select a Data Science demo to see how that function is used in real apps."
        )
    else:
        FEATURES[selected_function]()  # Streamlit function demo


