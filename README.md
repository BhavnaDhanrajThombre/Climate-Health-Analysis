# Climate Health Impact Analytics Dashboard

An interactive **Streamlit-based Climate Health Impact Analytics Dashboard** designed to explore relationships between climate conditions and public health outcomes.

The application combines interactive visual analytics with an AI-powered **Climate Health AI Assistant** that allows users to ask questions about the information currently displayed in the dashboard.

---

##  Dashboard Overview

The dashboard provides an interactive interface for exploring climate and public-health indicators across countries, regions, and years.

The dashboard is built using:

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **Plotly**

The application uses a dark analytical UI with interactive charts, KPI cards, filters, panels, and detailed analysis tabs.

The dashboard title is:

**Climate Health Impact Analytics Dashboard (2015–2025)**

with the subtitle:

**Understanding the Impact of Climate Change on Global Public Health**

---

##  Dashboard Features

### 1. Interactive Sidebar Filters

The dashboard provides a dynamic sidebar filtering system with:

* Country
* Region
* Income Level
* Year Range
* Month
* Temperature (°C)
* Air Quality Index (AQI)
* Reset Filters button

The filters are synchronized so that incompatible combinations are automatically handled.

All dashboard visualizations and calculations update according to the currently selected filters.

The dashboard also displays how many records and countries are currently represented after filtering.

---

##  Dashboard Tabs

The application contains seven analytical tabs.

###  1. Overview

The Overview tab provides a high-level summary of the currently filtered data.

It includes:

* Total Countries
* Population Covered
* Average Temperature
* Average AQI
* Average Respiratory Disease Rate
* Average Healthcare Access Index
* Top 10 Countries by Composite Risk
* Respiratory Disease Rate vs AQI
* Yearly Temperature Trend
* Disease Burden Composition by Region
* Key Insights
* Climate Health AI Assistant
  
---

###  2. Climate

The Climate tab provides detailed climate-related analysis and visualizations.

It focuses on indicators such as:

* Temperature
* Temperature anomalies
* Heatwave days
* Precipitation
* Air Quality Index
* Flood indicators
* Drought indicators

The visualizations allow users to explore climate patterns and variations across the selected filters.

---

###  3. Health

The Health tab focuses on public-health outcomes associated with the climate-health analysis.

It includes visualizations for:

* Respiratory Disease Rate
* Cardiovascular Mortality Rate
* Heat-Related Hospital Admissions
* Waterborne Disease Incidents
* Other health-related indicators

The tab also provides trend analysis across years and country-level comparisons.

---

###  4. Climate vs Health

This tab provides a deeper analysis of relationships between climate indicators and health outcomes.

Visualizations include:

* Temperature & Respiratory Disease Trend
* Heatwave Days vs Heat-Related Admissions
* Precipitation & Waterborne Disease Trend
* Climate Indicator Heatmap

These visualizations help users investigate associations between climate conditions and health indicators.

> Correlation is presented as an association and should not be interpreted as proof of causation.

---

###  5. Geographic

The Geographic tab provides country-level geographic analysis.

It includes a geographic visualization showing:

* Respiratory Disease Rate
* Population
* Country-level comparisons

This allows users to identify geographical patterns and regions with different levels of health burden.

---

###  6. Risk Analysis

The Risk Analysis tab provides a dedicated view of climate-health risk.

It includes:

* Global Risk Distribution
* Top 15 High-Risk Countries
* Risk by Region Treemap
* Detailed Risk Score table
* Risk Category classification

The risk methodology uses normalized averages of:

* Respiratory Disease Rate
* Cardiovascular Mortality Rate
* Air Quality Index
* Heat-Related Admissions

The resulting risk score is categorized into:

* **Low Risk**
* **Medium Risk**
* **High Risk**

using the 33rd and 66th percentile thresholds.

---

###  7. Correlation

The Correlation tab provides a detailed correlation analysis between selected climate and health indicators.

It includes:

* Correlation Matrix
* Correlation coefficients
* Executive Insights

The correlation analysis is calculated using the currently filtered data.

Correlation values range from **-1 to +1**.

A positive correlation indicates that two variables tend to increase together, while a negative correlation indicates that one tends to decrease as the other increases.

---

**Climate Health AI Assistant**

The chatbot is designed specifically to help users understand the analytical information displayed in the dashboard.

It is powered by:

* **Groq API**
* **Groq-hosted LLM**
* **Streamlit**
* **Pandas**
* **Python**
  
---

##  Anti-Hallucination Design

The chatbot uses a strict system prompt to ensure that responses remain grounded in the dashboard data.

The assistant is instructed to:

* Never fabricate statistics
* Never invent countries, years, or values
* Use computed values directly
* Not assume missing information
* Clearly distinguish averages, totals, rates, and indices
* Explain calculations when useful
* Treat correlation as association rather than causation
* Mention active filters when they materially affect the answer
* Redirect questions unrelated to supported dashboard topics

---

##  Risk Analysis Used by the Chatbot

The chatbot reuses the dashboard's existing risk methodology instead of creating a separate risk definition.

The risk score is based on normalized values of:

1. Respiratory Disease Rate
2. Cardiovascular Mortality Rate
3. Air Quality Index
4. Heat-Related Admissions

These normalized indicators are averaged into a single risk score.

Countries are then categorized using the 33rd and 66th percentile thresholds:

* Low Risk
* Medium Risk
* High Risk

This ensures that chatbot risk-related answers remain consistent with the Risk Analysis tab.

---

##  Correlation Analysis in the Chatbot

The chatbot can also use correlation information calculated from the currently filtered data.

The correlation snapshot identifies the strongest correlation among supported climate-health metrics and uses the same metric mapping as the dashboard's Correlation tab.

The chatbot is explicitly instructed not to describe correlation as causation.

For example:

*  "Temperature is positively associated with respiratory disease rate."
*  "Temperature causes respiratory disease."

---

#  Environment Configuration

The chatbot requires a Groq API key.

Create a `.env` file in the appropriate project directory and add:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

`GROQ_MODEL` is optional because the application has a default model configured.

**Do not commit your `.env` file or API key to GitHub.**

---

#  Dashboard–Chatbot Integration

The integration follows this flow:

```text
User selects dashboard filters
            ↓
       Filtered DataFrame
            ↓
    Local Python Analysis
            ↓
     Analytical Context
            ↓
       Groq AI Model
            ↓
     Natural Language Answer
            ↓
       Chatbot Interface
```

The important design principle is:

**Python calculates the dashboard statistics; the AI explains those calculated results.**

This helps maintain consistency between the visual dashboard and chatbot responses.

---

#  Chatbot Limitations

The Climate Health AI Assistant is an **analytical dashboard assistant**, not a medical professional.

It should not be used for:

* Medical diagnosis
* Individual medical advice
* Personal health decisions
* Clinical recommendations

The chatbot is limited to the information and analytical results available within the dashboard.

If the Groq API key is missing or an API request fails, the chatbot displays a warning instead of crashing the dashboard. The remaining dashboard functionality continues to work normally.

---

## 🛠️ Technology Stack

| Component             | Technology            |
| --------------------- | --------------------- |
| Programming Language  | Python                |
| Dashboard Framework   | Streamlit             |
| Data Processing       | Pandas, NumPy         |
| Visualization         | Plotly                |
| Generative AI         | Groq API              |
| AI Model              | `openai/gpt-oss-120b` |
| Environment Variables | python-dotenv         |
| Dashboard Styling     | Custom CSS            |

---

##  Application Architecture

```text
                   Climate Health Dashboard
                            │
             ┌──────────────┴──────────────┐
             │                             │
       Streamlit UI                 Sidebar Filters
             │                             │
             └──────────────┬──────────────┘
                            │
                     Filtered Data
                            │
              ┌─────────────┴─────────────┐
              │                           │
       Dashboard Analytics          AI Assistant
              │                           │
       ┌──────┼──────┐                    │
       │      │      │                    │
      KPIs  Charts  Risk/             Local Python
                  Correlation          Analysis
                                        │
                                   Analytical Context
                                        │
                                    Groq API / LLM
                                        │
                                   AI Explanation
```

---

##  Key Design Principle

> **The dashboard is the source of analytical truth, while the AI assistant acts as an explanatory interface over the dashboard's computed results.**

This ensures that chatbot responses remain aligned with the active filters, dashboard calculations, risk methodology, and correlation analysis.
