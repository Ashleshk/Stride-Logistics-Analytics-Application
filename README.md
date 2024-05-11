# Stride Logistics Analytics Application
A global e-commerce company aims to extract insights from consumer data by leveraging advanced machine learning. They seek to enhance understanding of customer behavior, preferences, and purchasing trends, empowering data-driven decision-making and bolstering business strategies.

 
> Stride in the context of the logistics domain refers to the spacing or distance between objects, such as packages, pallets, or containers, on a conveyor belt or in a warehouse. This concept is crucial in logistics and supply chain management to optimize storage, transportation, and inventory management. 
    -- source from google
 
 **BUSINESS GOALs Accomplishing : Optimizing Logistics Efficiency**

> 1. Analyze Stride Logistics' shipment data using advanced analytics tools.
> 
> 2. Provide valuable insights and data-driven decision-making support.
> 
> 3. Improve operational efficiency and reduce costs in logistics operations.
> 
> 4. Enhance customer satisfaction through data-driven optimizations.


## Problem Statement
The Stride logistics industry faces challenges in optimizing operational efficiency and enhancing customer satisfaction due to limited insights derived from shipment data.
### Why?
* Limited visibility into delivery time and customer rating for performance evaluation.
* Inefficient query response management impacting customer experience.
* Difficulty in determining the impact of product importance on delivery time and customer ratings.

### What I am trying to solve?
![goal](/image/goal.png)

## Development Plan

#### Phase 1: Exploratory Data Analysis
- Understand data dictionaries to grasp the meaning and structure of the dataset.
- Perform data checks, such as checking for missing values, data types, and descriptive statistics of categorical and numerical columns.
- Conduct univariate analysis to explore the distribution of individual variables.
- Perform bivariate analysis to investigate relationships between different variables.
- Summarize the findings and present them in graphical representations.
  
#### Phase 2: Feature Engineering, Correlation Analysis, and Machine Learning Modeling
- Introduce new features that are relevant to the problem and dataset.
- Evaluate the correlation between the target variable and the newly introduced features to select the most impactful ones.
- Explore and select multiple machine learning models and identify the best fit model based on evaluation metrics.
- Analyze the relationships between these factors and timely deliveries to provide further recommendations. For instance, you may identify certain weight ranges that are more prone to delays and suggest possible improvements in packaging or shipping methods.
  
#### Phase 3: Application Development
- Develop a Python Flask web application to enable user interaction.
- Users can upload an Excel sheet containing their data to the app.
- The application will perform analytical and machine learning operations on the uploaded sheet.
- Provide insights through exploratory data analysis, machine learning model results, accuracy measures, and business recommendations for current and future scenarios.
