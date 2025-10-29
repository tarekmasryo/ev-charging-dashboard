# ⚡ EV Charging Analytics Dashboard

[![Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-FF4B4B)](https://streamlit.io/)  
[![Made with ❤️ by Tarek Masryo](https://img.shields.io/badge/Made%20by-Tarek%20Masryo-blue)](https://github.com/tarekmasryo)

---

## 🎥 Live Preview

![Dashboard GIF](assets/Analytics.gif)

---

![Header](https://raw.githubusercontent.com/tarekmasryo/EV-Charging-Analytics/main/assets/header.png)


## 📌 Overview

Interactive dashboard built with **Streamlit, Plotly, and PyDeck** to explore the  
[Global EV Charging Stations & Models Dataset (2025)](https://github.com/tarekmasryo/Global-EV-Charging-Stations).

- 🌍 242k+ charging stations across 121 countries  
- ⚡ Filters for countries, power classes, and fast-DC  
- 📊 Insights, KPIs, and allocation optimizer  
- 🗺️ Interactive world map with clustering  

---

## 📊 Dashboard Preview

### Overview KPIs
![Overview](assets/overview.png)

### Advanced Map
![Map](assets/map.png)

### Insights (Impact vs Opportunity)
![Insights](assets/insights.png)

### Allocation Optimizer
![Optimizer](assets/optimizer.png)

---

## 🔑 Features

- **Filters** by country, power class, ports, and city  
- **KPIs**: total stations, ports, avg kW, fast-DC share  
- **Visuals**: donut charts, bar charts, Lorenz curve  
- **Optimizer**: simulate allocation of new fast-DC ports  
- **Compare mode**: selected vs global/regions  

---

## 🚀 Run Locally

Clone the repo and install requirements:

```bash
git clone https://github.com/tarekmasryo/ev-charging-analytics.git
cd ev-charging-analytics
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run EV-Charging-Analytics.py
```

---

## ☁️ Deploy on Streamlit Cloud

You can deploy directly to [Streamlit Cloud](https://streamlit.io/cloud).  
Make sure your **secrets / env** point to the dataset URL:

```toml
# .streamlit/secrets.toml
DATA_URL = "https://raw.githubusercontent.com/tarekmasryo/Global-EV-Charging-Stations/main/data/charging_stations_2025_world.csv"
```


---

## Related Repositories
- 📂 [Global EV Charging Dataset](https://github.com/tarekmasryo/global-ev-charging-dataset)
- 🔍 [EV Charging EDA](https://github.com/tarekmasryo/ev-charging-eda)


If you use this dashboard or dataset, please credit as:

> Global EV Charging Dashboard and Dataset by **Tarek Masryo**.\
> Code licensed under Apache 2.0 . Data licensed under CC BY 4.0.
