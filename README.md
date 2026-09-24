# India City Air Quality Analytics & Next-Day AQI Forecasting

An interactive analytics dashboard that turns five years of daily air quality readings from Indian cities into clear insights, a next-day AQI forecast, and recommended actions for city authorities.

Built for the **AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026** (BharatCares).

## Problem statement

Air pollution is one of India's most serious public health risks, but raw sensor readings do not tell decision-makers where, when, and why air quality becomes dangerous. This project answers four questions:

1. Which cities have the worst air, and how often is it unsafe?
2. When does pollution peak (seasons, months, years)?
3. Which pollutants drive the Air Quality Index (AQI)?
4. Can we forecast tomorrow's AQI early enough to issue health alerts?

## Dataset

- **Name:** Air Quality Data in India (2015–2020), file `city_day.csv`
- **Source:** Central Pollution Control Board (CPCB) data, published on Kaggle
- **Link:** https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
- **Contents:** daily readings for 26 Indian cities: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI and AQI category

## Features

| Page                  | What it shows                                                                                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Executive overview    | KPIs (average AQI, % of days Poor or worse, Severe days, most and least polluted city), city ranking, AQI category distribution, data cleaning summary              |
| Trends & seasons      | Monthly AQI trend, month × year heatmap, seasonal comparison, 2020 lockdown impact                                                                                  |
| Pollution drivers     | Pollutant correlation with AQI, seasonal pollutant levels, city pollution profiles                                                                                  |
| Next-day AQI forecast | Random forest model, evaluation against a "tomorrow = today" baseline, actual vs forecast chart, feature importance, interactive forecast form with health advisory |
| Insights & actions    | Auto-generated key facts, risks, opportunities and recommended actions                                                                                              |

Sidebar filters (cities and years) apply to all analysis pages.

## Methodology

1. **Data cleaning:** date parsing, numeric conversion, negative readings treated as sensor errors, duplicate removal, pollutant gaps of up to 3 days filled by interpolation within each city. AQI itself is never imputed; the AQI category is recomputed from CPCB breakpoints.
2. **Exploratory analysis:** city, time and seasonal patterns; the 2020 lockdown used as a natural experiment.
3. **Driver analysis:** correlation of each pollutant with AQI and seasonal pollutant behaviour.
4. **Forecasting:** a Random Forest Regressor predicts the next day's AQI from today's pollutant levels, today's AQI, the 7-day average AQI, month and city. The model is trained on data before July 2019 and tested on later, unseen dates, and compared with a persistence baseline.
5. **Business intelligence:** findings are translated into risks, opportunities and recommended actions.

## Key results

- Average AQI of 166 (Moderate) across 26 cities; 26% of city-days were Poor or worse
- Winter air is about 1.9 times as polluted as monsoon air; November is the worst month
- AQI fell 39% during the 2020 lockdown compared with the same weeks in 2019
- The next-day forecast has an average error of 18.6 AQI points, 16% better than assuming tomorrow equals today, and catches 88% of days above AQI 200

## Tech stack

Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit

## How to run

Use Python 3.10–3.13.

```bash
git clone https://github.com/siddhipandhere/India-Air-Quality-Analytics.git
cd India-Air-Quality-Analytics
pip install -r requirements.txt
```

Download `city_day.csv` from the Kaggle link above and place it in the same folder as the code file (the other CSV files in the dataset are not needed), then run:

```bash
python -m streamlit run Siddhi_Pandhere_IndiaAirQuality.py
```

The dashboard opens at http://localhost:8501. If `city_day.csv` is not in the folder, the app lets you upload it from the sidebar.

## Project structure

```
├── Siddhi_Pandhere_IndiaAirQuality.py   # complete application: cleaning, analysis, model and dashboard
├── requirements.txt              # Python dependencies
├── README.md                     # this file
├── Siddhi_Pandhere_ProjectReport.docx   # project report
└── .streamlit/
    └── config.toml               # keeps the dashboard in its light colour theme
```

## Author

Siddhi Pandhere · Pillai College of Engineering, New Panvel · AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
