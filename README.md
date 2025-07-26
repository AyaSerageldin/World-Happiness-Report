# 🌍 World Happiness Prediction

This project explores global happiness factors using data from the United Nations **World Happiness Report**. It applies predictive analytics and machine learning to understand what drives happiness in different countries and builds models to predict happiness scores based on key economic, social, and political indicators.

## 🧠 Problem Formulation

On July 12, 2012, the United Nations declared March 20th as the **International Day of Happiness**, a resolution initiated by Bhutan — a country known for prioritizing Gross National Happiness over Gross Domestic Product. Since then, the UN has released the **World Happiness Report** annually, measuring happiness across countries using indicators that reflect societal well-being.

This project aims to analyze these indicators, identify patterns in global happiness, and build predictive models that can help policymakers understand which factors contribute most to the well-being of their populations. By applying data science techniques, we strive to uncover the “happiness equation” and promote more inclusive and equitable societies.

## 🛠 Tools & Technologies

- **Programming:** Python  
- **Libraries:** pandas · NumPy · seaborn · matplotlib · scikit-learn  
- **Notebook Environment:** Jupyter Notebook  
- **Modeling Techniques:** Linear Regression · Random Forest · KMeans Clustering

## 📊 Data Source

- United Nations World Happiness Report  
  [https://worldhappiness.report](https://worldhappiness.report)

## 📈 Key Features

- In-depth exploratory data analysis (EDA)
- Data preprocessing & cleaning of null or incomplete entries
- Predictive modeling to estimate happiness scores
- Clustering to group countries with similar happiness profiles
- Insights for governments and organizations to guide policy decisions

## 🧪 Project Reflections & Challenges

Through this project, we successfully:
- Identified key happiness factors such as GDP per capita, social support, and perceptions of corruption.
- Built a regression model to predict happiness scores.
- Developed clustering models to identify countries with similar profiles.

Challenges included:
- Handling null values for specific countries, requiring data wrangling to approximate missing entries.
- High dimensionality, which added complexity to model selection and clustering performance.
- Limitations in clustering accuracy due to analyzing data from a single year — future versions should consider average values across multiple years.
- Opportunities to enhance clustering by testing other algorithms such as **hierarchical clustering**.

## 🚀 Future Improvements

- Integrate a dashboard for interactive visualizations (e.g., Streamlit or Tableau)
- Evaluate more machine learning models (e.g., Gradient Boosting, XGBoost)
- Extend the dataset to analyze trends across multiple years
- Test alternative clustering techniques and validate the results
