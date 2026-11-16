# 🌿 LifeStat – Personal Lifestyle Statistical Analyzer

LifeStat is an interactive Streamlit web application that helps you understand your everyday habits
(sleep, exercise, water intake, screen time, study hours, energy and mood) using **Probability and
Statistics**. It is a 3rd-semester engineering Mathematics project.

The mathematics works behind the scenes; the app itself uses everyday names such as
*Sleep & Energy Insights* and *Habit Connections*.

---

## Existing systems vs LifeStat

| Area | Conventional lifestyle trackers (typical focus) | LifeStat (proposed system) |
|---|---|---|
| Main purpose | Basic activity tracking and habit monitoring | Statistical understanding of lifestyle data |
| Summaries | Daily/weekly/monthly averages, general summaries | Averages plus variance and standard deviation |
| Patterns | Progress over time | Relationship-based analysis |
| Chances of events | Usually not the focus | Probability-based insights |
| Spread of values | Usually shown as averages | Distribution analysis |
| Relationships | Usually not the focus | Regression lines and Pearson correlation |
| Charts | Basic progress charts | Interactive statistical visualisations |

LifeStat does not claim these features are missing from every existing app. Its distinct contribution is
the **combination** of personal tracking with five mathematical methods in one interactive application.

---

## The five mathematical modules

| App page | Mathematical topic | Main formulas |
|---|---|---|
| Sleep & Energy Insights | Joint probability | P(A ∩ B) = days with both ÷ total days |
| Energy Patterns | Discrete probability distribution | E(X) = Σ x·P(X = x); Var(X) = E(X²) − [E(X)]² |
| Sleep Duration Insights | Continuous probability distribution | Mean, sample variance (n − 1), standard deviation, density estimate, cumulative distribution, range probability |
| Lifestyle Relationships | Regression lines | Y − Ȳ = b_yx (X − X̄); X − X̄ = b_xy (Y − Ȳ); R² |
| Habit Connections | Pearson correlation | r = Σ(x − x̄)(y − ȳ) ÷ √[Σ(x − x̄)² Σ(y − ȳ)²] |

Design decisions worth explaining in your viva:

* **Joint probability** only counts days (no conditional probability or Bayes' theorem).
* **Discrete distribution** treats the whole-number energy rating (1–10) as the random variable X.
* **Continuous distribution** uses the data's own (empirical) distribution first. A fitted normal curve is
  optional and shown with its assumptions and a Shapiro-Wilk check. It is never assumed automatically.
* **Regression** gives both lines, because each one predicts a different variable. A useful check shown in
  the app is R² = b_yx × b_xy.
* **Correlation** uses Pearson's r only (no Spearman).

---

## Features

* Overview dashboard with metric cards and interactive Plotly charts
* Manual daily entry with validation, plus CSV upload with friendly error messages
* Downloadable CSV template and a built-in 80-day illustrative sample dataset
* Record viewing, deleting, resetting and CSV export; all analyses recalculate automatically
* Downloadable report (TXT) and key-results table (CSV)
* Helpful messages for empty data, too few records, missing values, duplicate dates and
  zero-variance data (no crashes)

---

## Project structure

```
LifeStat/
├── app.py                     # Streamlit interface (all pages)
├── stats_engine.py            # The five mathematical modules
├── data_utils.py              # Data checking, sample loading, CSV helpers
├── report_builder.py          # Report text and summary table
├── requirements.txt           # Libraries for Streamlit Community Cloud
├── README.md                  # This file
├── .gitignore
├── .streamlit/
│   └── config.toml            # Green colour theme
├── data/
│   └── sample_lifestyle_data.csv   # 80 days of illustrative sample data
└── tests/
    └── test_lifestat.py       # Checks the mathematics and data handling
```

---

## Run it on your own computer (optional)

1. Install Python 3.10 or newer.
2. Open a terminal in the `LifeStat` folder.
3. Install the libraries: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`
5. (Optional) Check the mathematics: `python tests/test_lifestat.py`

---

## Deploy online with GitHub and Streamlit Community Cloud

1. **Download and extract.** Unzip `LifeStat.zip`. You should see the `LifeStat` folder with `app.py` inside.
2. **Create a GitHub repository.** Sign in at github.com, click **New repository**, name it
   `LifeStat`, choose **Public**, and click **Create repository**.
3. **Upload the files.** On the new repository page click **uploading an existing file**. Open the
   extracted `LifeStat` folder and drag **everything inside it** (including the `data`, `tests` and
   `.streamlit` folders and the `.gitignore` file) into the browser, then click **Commit changes**.
   `app.py` must be at the top level of the repository, not inside another folder.
   *If your computer hides files starting with a dot, upload `.gitignore` and `.streamlit/config.toml`
   using **Add file → Create new file**, or skip them; the app works without them.*
4. **Open Streamlit Community Cloud.** Go to share.streamlit.io and sign in with your GitHub account.
5. **Connect the repository.** Click **Create app**, choose to deploy from an existing GitHub repository,
   and select your `LifeStat` repository and the `main` branch.
6. **Select the main file.** Set **Main file path** to `app.py`.
7. **Deploy.** Click **Deploy**. After a few minutes your live link (ending in `.streamlit.app`) opens
   the application. Share that link in your presentation.

---

## Data persistence (important)

LifeStat stores records in Streamlit **session state**, which lives in the browser session only.

* Records are **not** saved permanently. Refreshing the page or starting a new visit returns to the sample data.
* Streamlit Community Cloud apps can restart or sleep, and their file system is not a reliable place
  for user data, so LifeStat deliberately does not write records to the server.
* To keep your data, use **My Records → Download records as CSV** and upload that file next time.
* Each visitor has a separate session, so users never see each other's records.

---

## Limitations

* Statistics describe only the records provided. Small datasets (under about 20 days) give rough estimates.
* Regression and correlation show relationships, **not cause and effect**.
* Pearson's r detects straight-line links only.
* The sample data is generated for demonstration and is **not** real personal data.
* The report is plain text (TXT) plus a CSV table, chosen for reliable deployment without extra libraries.
* LifeStat is an educational project and does not give medical advice.

---

## Technology

Python · Streamlit · Pandas · NumPy · SciPy · Plotly. No paid services, API keys, AI or machine learning.
