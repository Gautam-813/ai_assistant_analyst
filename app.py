
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import datetime as dt
import math
import requests
import json

try:
    from groq import Groq
except ImportError:
    Groq = None

st.set_page_config(page_title="AI Quant Research Copilot", layout="wide")

# Simple light theme
st.markdown(
    """
    <style>
    .main { background-color: #ffffff; color: #000000; }
    [data-testid="stSidebar"] { background-color: #f1f3f5; }
    h1, h2, h3, p, span, label { color: #000000 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Password protection
def check_password():
    """Check if password is correct and store in session state."""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    
    if not st.session_state.password_correct:
        st.title("🔐 Authentication Required")
        st.markdown("Enter the password to access the AI Quant Research Copilot dashboard.")
        
        password_input = st.text_input(
            "Password",
            type="password",
            key="password_input",
            placeholder="Enter password..."
        )
        
        if st.button("Unlock Dashboard"):
            stored_password = st.secrets.get("PASSWORD", "")
            if password_input == stored_password:
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
        return False
    return True

# Check password before showing dashboard
if not check_password():
    st.stop()


@st.cache_data
def load_asset_data(symbol: str):
    """Load raw 1m data for a symbol, limited to recent history for speed."""
    import os

    filenames = {
        "GOLD": "XAUUSD_M1_Data.parquet",
        "DXY": "DXY_M1_Data.parquet",
        "EURUSD": "EURUSD_M1_Data.parquet",
    }
    fname = filenames.get(symbol)
    if not fname:
        return None

    possible_paths = [
        fname,
        f"atr_analysis_data/{fname}",
        r"d:\date-wise\03-03-2026\atr_analysis_data\\" + fname,
    ]
    file_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if file_path is None:
        return None

    df = pd.read_parquet(file_path)
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.sort_values("Time")

    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    return df


@st.cache_data
def get_available_models(gemini_key, groq_key=None, openrouter_key=None, bytez_key=None, nvidia_key=None):
    models = []

    try:
        if gemini_key:
            genai.configure(api_key=gemini_key)
            gemini_models = [
                m.name
                for m in genai.list_models()
                if "generateContent" in getattr(m, "supported_generation_methods", [])
            ]
            models.extend([m for m in gemini_models if "gemini" in m.lower()])
    except Exception:
        models.extend(["models/gemini-1.5-flash", "models/gemini-2.0-flash"])

    if groq_key and Groq:
        models.extend(
            [
                "groq/llama-3.3-70b-versatile",
                "groq/llama-3.1-8b-instant",
            ]
        )

    if openrouter_key:
        models.extend(
            [
                "openrouter/qwen/qwen-2.5-72b-instruct:free",
                "openrouter/meta-llama/llama-3.3-70b-instruct:free",
            ]
        )

    # Bytez: we can't query models dynamically yet, so expose a small curated set
    if bytez_key:
        models.extend(
            [
                "bytez/Qwen/Qwen3-4B",
                "bytez/Qwen/Qwen2.5-7B-Instruct",
            ]
        )

    # NVIDIA: add NVIDIA NIM models
    if nvidia_key:
        models.extend(
            [
                "nvidia/meta-llama/llama-3.1-405b-instruct",
                "nvidia/meta-llama/llama-3.1-70b-instruct",
                "nvidia/meta-llama/llama-3.2-3b-instruct",
            ]
        )

    models = sorted(set(models))
    return models


def prepare_asset_views(df: pd.DataFrame, max_days: int):
    """Trim to recent history and build a lightweight daily view."""
    if df is None or df.empty or "Time" not in df.columns:
        return None, None

    latest_time = df["Time"].max()
    cutoff = latest_time - pd.Timedelta(days=max_days)
    trimmed = df[df["Time"] >= cutoff].copy()

    daily = (
        trimmed.set_index("Time")[["Open", "High", "Low", "Close"]]
        .resample("1D")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
        .reset_index()
    )
    daily["Return"] = daily["Close"].pct_change()

    return trimmed, daily


# Sidebar controls
st.sidebar.header("⚙️ AI Engine Settings")

with st.sidebar.expander("🔑 AI API Keys", expanded=True):
    st.caption("Keys live only in memory for this session.")
    u_gemini_key = st.text_input("Gemini API Key", type="password")
    u_groq_key = st.text_input("Groq API Key", type="password")
    u_or_key = st.text_input("OpenRouter API Key", type="password")
    u_bytez_key = st.text_input("Bytez API Key", type="password")
    u_nvidia_key = st.text_input("NVIDIA API Key", type="password")

    final_gemini_key = u_gemini_key or st.secrets.get("GEMINI_API_KEY", "")
    final_groq_key = u_groq_key or st.secrets.get("GROQ_API_KEY", "")
    final_or_key = u_or_key or st.secrets.get("OPEN_ROUTER", "")
    final_bytez_key = u_bytez_key or st.secrets.get("Bytez", "")
    final_nvidia_key = u_nvidia_key or st.secrets.get("NVIDIA_API_KEY", "")

    if final_gemini_key:
        st.success("Gemini: Connected")
    if final_groq_key:
        st.success("Groq: Connected")
    if final_or_key:
        st.success("OpenRouter: Connected")
    if final_bytez_key:
        st.success("Bytez: Connected")
    if final_nvidia_key:
        st.success("NVIDIA: Connected")

st.sidebar.subheader("📊 Data Window")
max_days = st.sidebar.slider("Use last N days of data", 30, 365, 180, step=15)

st.sidebar.subheader("🌍 Assets exposed to AI")
ai_symbols = []
if st.sidebar.checkbox("GOLD (XAUUSD)", value=True):
    ai_symbols.append("GOLD")
if st.sidebar.checkbox("DXY (US Dollar Index)", value=False):
    ai_symbols.append("DXY")
if st.sidebar.checkbox("EURUSD", value=False):
    ai_symbols.append("EURUSD")


# Load data once
gold_raw = load_asset_data("GOLD")
dxy_raw = load_asset_data("DXY")
eur_raw = load_asset_data("EURUSD")

if gold_raw is None:
    st.error("Gold data (XAUUSD_M1_Data.parquet) not found. The AI needs at least GOLD data.")
    st.stop()

gold_raw, gold_daily = prepare_asset_views(gold_raw, max_days)
if dxy_raw is not None:
    dxy_raw, dxy_daily = prepare_asset_views(dxy_raw, max_days)
else:
    dxy_daily = None
if eur_raw is not None:
    eur_raw, eur_daily = prepare_asset_views(eur_raw, max_days)
else:
    eur_daily = None


# Session state
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0
if "messages" not in st.session_state:
    st.session_state.messages = []


# Main layout
st.title("🤖 AI Quant Research Copilot")
st.markdown(
    "Ask any quant or trading question about your **GOLD / DXY / EURUSD** data. "
    "The AI will write and execute pandas/Plotly code on the fly."
)

# Quick data summary for user context (cheap to compute)
with st.expander("📁 Data overview", expanded=False):
    def _summary(name, raw_df, daily_df):
        if raw_df is None or raw_df.empty:
            st.write(f"{name}: not available")
            return
        start = raw_df["Time"].min()
        end = raw_df["Time"].max()
        st.write(
            f"**{name}** — rows: {len(raw_df):,}, from {start} to {end}, daily points: {0 if daily_df is None else len(daily_df):,}"
        )

    _summary("GOLD", gold_raw, gold_daily)
    _summary("DXY", dxy_raw, dxy_daily)
    _summary("EURUSD", eur_raw, eur_daily)


# Guard: require at least one provider
if not (final_gemini_key or final_groq_key or final_or_key or final_bytez_key or final_nvidia_key):
    st.warning("Provide at least one API key (Gemini, Groq, OpenRouter, Bytez, or NVIDIA) in the sidebar to use the AI.")
    st.stop()


# Resolve models
available_models = get_available_models(final_gemini_key, final_groq_key, final_or_key, final_bytez_key, final_nvidia_key)
if not available_models:
    st.error("No models available. Check your API keys.")
    st.stop()

c_ai1, c_ai2 = st.columns([1, 2])
with c_ai1:
    default_ix = 0
    if "models/gemini-2.0-flash" in available_models:
        default_ix = available_models.index("models/gemini-2.0-flash")
    elif "groq/llama-3.3-70b-versatile" in available_models:
        default_ix = available_models.index("groq/llama-3.3-70b-versatile")

    sel_model_name = st.selectbox("AI Model", options=available_models, index=default_ix)
    if sel_model_name.startswith("groq"):
        provider = "Groq"
    elif sel_model_name.startswith("openrouter"):
        provider = "OpenRouter"
    elif sel_model_name.startswith("bytez"):
        provider = "Bytez"
    elif sel_model_name.startswith("nvidia"):
        provider = "NVIDIA"
    else:
        provider = "Google Gemini"
    st.caption(f"Provider: {provider}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with c_ai2:
    st.info(
        "You can ask multi-step questions like:\n\n"
        "- \"Show the volatility regime for GOLD over the last 90 days.\"\n"
        "- \"Compare daily returns of GOLD vs DXY and plot a scatter.\"\n"
        "- \"Build a simple mean-reversion backtest on GOLD daily closes.\""
    )


# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "code" in message:
            with st.expander("View generated code"):
                st.code(message["code"], language="python")
        if "result" in message:
            res = message["result"]
            if hasattr(res, "to_dict") and "data" in res.to_dict():
                try:
                    res.update_xaxes(type="category")
                except Exception:
                    pass
                st.plotly_chart(res, use_container_width=True)
            else:
                st.write(res)


user_input = st.chat_input("Ask a quant question...")

if user_input:
    if not ai_symbols:
        st.warning("Select at least one asset in the sidebar for the AI to analyze.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build AI execution environment
    agg_str, raw_str = [], []
    safe_globals = {
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "dt": dt,
        "math": math,
        "result": None,
    }

    if "GOLD" in ai_symbols:
        if gold_daily is not None:
            agg_str.append(f"- `gold_daily` (cols: {gold_daily.columns.tolist()})")
            safe_globals["gold_daily"] = gold_daily
        agg_str.append("- `gold_info` (helper dict with meta)")
        safe_globals["gold_info"] = {
            "rows": len(gold_raw),
            "start": gold_raw["Time"].min(),
            "end": gold_raw["Time"].max(),
        }
        raw_str.append(f"- `gold_raw` (cols: {gold_raw.columns.tolist()})")
        safe_globals["gold_raw"] = gold_raw

    if "DXY" in ai_symbols and dxy_raw is not None:
        if dxy_daily is not None:
            agg_str.append(f"- `dxy_daily` (cols: {dxy_daily.columns.tolist()})")
            safe_globals["dxy_daily"] = dxy_daily
        raw_str.append(f"- `dxy_raw` (cols: {dxy_raw.columns.tolist()})")
        safe_globals["dxy_raw"] = dxy_raw

    if "EURUSD" in ai_symbols and eur_raw is not None:
        if eur_daily is not None:
            agg_str.append(f"- `eur_daily` (cols: {eur_daily.columns.tolist()})")
            safe_globals["eur_daily"] = eur_daily
        raw_str.append(f"- `eur_raw` (cols: {eur_raw.columns.tolist()})")
        safe_globals["eur_raw"] = eur_raw

    agg_text = "\n                ".join(agg_str) if agg_str else "None selected."
    raw_text = "\n                ".join(raw_str) if raw_str else "None selected."

    # Limit history length and code size to keep prompts small and fast
    context_history = ""
    history_tail = st.session_state.messages[-6:-1]  # last 5 turns before this one
    for msg in history_tail:
        text = msg["content"]
        if len(text) > 1200:
            text = text[:1200] + "...[truncated]"
        context_history += f"{msg['role'].upper()}: {text}\n"
        if "code" in msg:
            code_snip = msg["code"]
            if len(code_snip) > 1500:
                code_snip = code_snip[:1500] + "...[truncated]"
            context_history += f"PREVIOUS CODE GENERATED (TRUNCATED):\n{code_snip}\n"

    base_prompt = f"""
You are a senior quantitative researcher. You write fast, production-quality pandas and Plotly code.

You have access ONLY to these in-memory objects:
1) Aggregated daily views:
{agg_text}
2) High-frequency history:
{raw_text}

Conversation history (truncated):
{context_history}

New user request:
{user_input}

Requirements:
- Return ONLY raw executable Python code (no markdown, no explanations).
- Do NOT reload data from disk; use the provided DataFrames.
- Prefer using the daily DataFrames for heavy work; use raw 1-minute data only when truly needed.
- Avoid huge intermediate DataFrames; down-sample or slice recent ranges when plotting.
- Store your final answer in a variable named `result`.
- For visualizations, assign the Plotly figure (or matplotlib axis) to `result`.
"""

    max_retries = 2  # keep retries small for latency
    current_attempt = 1
    final_code = ""
    final_result = None
    usage_text = ""
    error_feedback = ""

    with st.chat_message("assistant"):
        with st.status("Quant engine running...", expanded=False) as status:
            while current_attempt <= max_retries:
                if error_feedback:
                    status.update(
                        label=f"Self-correcting (attempt {current_attempt})...",
                        state="running",
                    )
                    prompt_to_send = (
                        base_prompt
                        + f"\nPrevious execution error (fix this): `{error_feedback}`\nReturn code only."
                    )
                else:
                    prompt_to_send = base_prompt

                try:
                    if sel_model_name.startswith("groq"):
                        client = Groq(api_key=final_groq_key)
                        chat_completion = client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt_to_send}],
                            model=sel_model_name.split("/")[-1],
                        )
                        gen_text = chat_completion.choices[0].message.content.strip()
                        usage = chat_completion.usage
                        st.session_state.total_input_tokens += usage.prompt_tokens
                        st.session_state.total_output_tokens += usage.completion_tokens
                        st.session_state.total_cost += (
                            usage.total_tokens / 1_000_000
                        ) * 0.15
                        usage_text = (
                            f"Input: {usage.prompt_tokens} | Output: {usage.completion_tokens}"
                        )
                    elif sel_model_name.startswith("openrouter"):
                        or_model = sel_model_name.replace("openrouter/", "")
                        headers = {
                            "Authorization": f"Bearer {final_or_key}",
                            "Content-Type": "application/json",
                        }
                        payload = {
                            "model": or_model,
                            "messages": [{"role": "user", "content": prompt_to_send}],
                        }
                        response = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers=headers,
                            data=json.dumps(payload),
                        )
                        res_json = response.json()
                        if "choices" not in res_json:
                            raise ValueError(
                                f"OpenRouter Error: {res_json.get('error', 'Unknown Error')}"
                            )
                        gen_text = res_json["choices"][0]["message"]["content"].strip()
                        usage = res_json.get(
                            "usage",
                            {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                            },
                        )
                        st.session_state.total_input_tokens += usage["prompt_tokens"]
                        st.session_state.total_output_tokens += usage[
                            "completion_tokens"
                        ]
                        st.session_state.total_cost += (
                            usage["total_tokens"] / 1_000_000
                        ) * 0.10
                        usage_text = (
                            f"Input: {usage['prompt_tokens']} | Output: {usage['completion_tokens']}"
                        )
                    elif sel_model_name.startswith("bytez"):
                        # Bytez native HTTP API
                        bytez_model = sel_model_name.replace("bytez/", "")
                        headers = {
                            "Authorization": final_bytez_key,
                            "Content-Type": "application/json",
                        }
                        payload = {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": prompt_to_send,
                                }
                            ]
                        }
                        response = requests.post(
                            f"https://api.bytez.com/models/v2/{bytez_model}",
                            headers=headers,
                            data=json.dumps(payload),
                        )
                        res_json = response.json()
                        if "output" not in res_json:
                            raise ValueError(
                                f"Bytez Error: {res_json.get('error', 'No output field in response')}"
                            )
                        output = res_json["output"]
                        if isinstance(output, str):
                            gen_text = output.strip()
                        else:
                            # Some models return list/structured output; fall back to JSON string
                            gen_text = json.dumps(output)
                        # Bytez docs don't yet expose token usage in this endpoint
                        usage_text = "Bytez usage: not reported"
                    elif sel_model_name.startswith("nvidia"):
                        # NVIDIA NIM API
                        nvidia_model = sel_model_name.replace("nvidia/", "")
                        headers = {
                            "Authorization": f"Bearer {final_nvidia_key}",
                            "Content-Type": "application/json",
                        }
                        payload = {
                            "model": nvidia_model,
                            "messages": [{"role": "user", "content": prompt_to_send}],
                            "max_tokens": 4096,
                            "temperature": 0.3,
                        }
                        response = requests.post(
                            "https://integrate.api.nvidia.com/v1/chat/completions",
                            headers=headers,
                            data=json.dumps(payload),
                        )
                        res_json = response.json()
                        if "choices" not in res_json:
                            raise ValueError(
                                f"NVIDIA Error: {res_json.get('error', 'Unknown Error')}"
                            )
                        gen_text = res_json["choices"][0]["message"]["content"].strip()
                        usage = res_json.get(
                            "usage",
                            {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                            },
                        )
                        st.session_state.total_input_tokens += usage["prompt_tokens"]
                        st.session_state.total_output_tokens += usage[
                            "completion_tokens"
                        ]
                        st.session_state.total_cost += (
                            usage["total_tokens"] / 1_000_000
                        ) * 0.10
                        usage_text = (
                            f"Input: {usage['prompt_tokens']} | Output: {usage['completion_tokens']}"
                        )
                    else:
                        genai.configure(api_key=final_gemini_key)
                        model = genai.GenerativeModel(sel_model_name)
                        response = model.generate_content(prompt_to_send)
                        gen_text = response.text.strip()
                        usage = response.usage_metadata
                        st.session_state.total_input_tokens += usage.prompt_token_count
                        st.session_state.total_output_tokens += (
                            usage.candidates_token_count
                        )
                        st.session_state.total_cost += (
                            usage.total_token_count / 1_000_000
                        ) * 0.075
                        usage_text = (
                            f"Input: {usage.prompt_token_count} | Output: {usage.candidates_token_count}"
                        )

                    if "```python" in gen_text:
                        gen_text = gen_text.split("```python")[-1].split("```")[0].strip()
                    elif "```" in gen_text:
                        gen_text = gen_text.split("```")[-1].split("```")[0].strip()

                    safe_globals["result"] = None
                    exec(gen_text, safe_globals)
                    final_result = safe_globals.get("result")
                    if final_result is None:
                        raise ValueError("`result` variable was not assigned.")

                    final_code = gen_text
                    status.update(label="Analysis complete", state="complete")
                    break
                except Exception as e:
                    error_feedback = str(e)
                    current_attempt += 1

            if not final_code:
                st.error(f"Quant engine failed: {error_feedback}")

        if final_code:
            with st.expander("View generated code"):
                st.code(final_code, language="python")
            if hasattr(final_result, "to_dict") and "data" in final_result.to_dict():
                try:
                    final_result.update_xaxes(type="category")
                except Exception:
                    pass
                st.plotly_chart(final_result, use_container_width=True)
            else:
                st.write(final_result)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Analysis complete.",
                    "code": final_code,
                    "result": final_result,
                }
            )
            if usage_text:
                st.caption(f"Token usage (last call): {usage_text}")


st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Session tokens",
        f"{st.session_state.total_input_tokens + st.session_state.total_output_tokens:,}",
    )
with col2:
    st.metric(
        "Input / Output tokens",
        f"{st.session_state.total_input_tokens:,} / {st.session_state.total_output_tokens:,}",
    )
with col3:
    st.metric("Estimated session cost (USD)", f"{st.session_state.total_cost:.5f}")

