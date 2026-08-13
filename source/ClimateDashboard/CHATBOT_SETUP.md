# Climate Health AI Assistant — Setup

## 1. Files changed
- `source/ClimateDashboard/dashboard.py` — 2 small additions only:
  - one new import line: `from chatbot import render_climate_health_chatbot`
  - one new block at the very bottom of the `📊 Overview` tab (after `Key Insights`,
    inside the same `with tabs[0]:` block) that opens a `panel_open("Climate Health AI
    Assistant", "purple")` panel and calls `render_climate_health_chatbot(df)`.
  Nothing else in the file was touched — same charts, same CSS, same tabs, same filters.
- `source/ClimateDashboard/chatbot.py` — **new file**, all chatbot logic lives here.

## 2. Dependencies to add to `requirements.txt`
```
groq>=0.11.0
python-dotenv>=1.0.1
```

## 3. `.env` variables
```
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```
`GROQ_MODEL` is optional — it defaults to `openai/gpt-oss-120b`, which is Groq's
current recommended flagship open model. Change it in `.env` if you'd rather use
`openai/gpt-oss-20b` (cheaper/faster) or another Groq-hosted model.

## 4. How it works
1. `render_climate_health_chatbot(df)` is called with the dashboard's already-filtered
   `df` (the same one from `df = apply_filters(df_raw)`), so it's always in sync with
   whatever the sidebar filters currently show.
2. `build_data_context(df)` computes everything locally with pandas first: row/country/
   region counts, per-metric mean/min/max/median/std, top-5 countries by AQI/respiratory
   rate/heat admissions, region breakdowns, year-over-year change, a risk analysis (reusing
   the dashboard's own `compute_risk()` — same normalization + 33rd/66th percentile logic,
   not a new definition), and a correlation snapshot (reusing the dashboard's `CORR_COLS`).
3. That compact context (a few KB, not the raw CSV) plus a strict anti-hallucination
   system prompt and the conversation history are sent to Groq's Chat Completions API
   (`client.chat.completions.create`).
4. The model is instructed to answer only from the supplied computed numbers, never
   invent stats, and describe correlations as "associated with" rather than "causes."
5. Chat state lives in `st.session_state.chat_messages`, capped at the last 12 messages.
6. If `GROQ_API_KEY` is missing, or the API call fails/times out/returns empty, the
   assistant shows a friendly warning instead of crashing — the rest of the dashboard
   (all 7 tabs) keeps working normally either way.

## 5. Placement confirmation
The assistant renders **only** inside `📊 Overview`, physically after the KPI row,
the three top charts, Disease Burden by Region, and Key Insights — nothing was moved,
resized, or restyled, and no other tab was touched.

## 6. Suggested questions included
- "What are the top 5 high-risk countries?"
- "Which region has the highest AQI?"
- "How is temperature related to respiratory disease?"
- "Summarize the current filtered data."

Tested locally (with a stubbed Streamlit session) against `climate_health_cleaned.csv`
to confirm the analytical context builds correctly (14,100 rows, 25 countries, 8 regions,
stats, top-N, region breakdowns, YoY, risk, correlation all populate as expected).
