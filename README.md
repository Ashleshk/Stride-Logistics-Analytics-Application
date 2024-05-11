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



## Live WebApp on Wix

> Link - *Working on it, please keep reading*
> Demo Recording : 
https://github.com/Ashleshk/Stride-Logistics-Analytics-Application/blob/main/DemoVideo.mp4


# Results
## Exploratory Data Analysis
### Numerical Columns:
Prior Purchases and Discount Offered columns have minor skewness.
Cost of the Product column follows an almost normal distribution.
Customer ratings and customer care calls have a balanced distribution.
Weight in grams follows a u-shaped uncertain distribution.
Reached on Time column has a binary distribution.
### Categorical Columns:
Deliveries for Warehouse Block F have the highest number not delivered on time.
Shipments made by ships tend to run late.
Delivery delays are common for products with low and medium importance.
Female and male customers show similar delivery behavior.
### Bivariate Analysis:
Mode of Shipment:
Ship mode has the highest number of products not delivered on time.
Product Importance:
Deliveries are not on time for products of all importance levels, particularly high and low importance.
Warehouse Block:
Warehouse Block F has the highest number of products not delivered on time.
Weight Category:
Deliveries are generally on time for all weight categories except "wcat6" and "wcat7."
Discount Category:
The majority of products fall into "dcat1" with varied delivery results.

## Feature Engineering ( Introduced Total 20 features)
- Mode_of_Shipment_encoded: Encoded values of the "Mode_of_Shipment" column, mapping "Ship" to 1, "Flight" to 2, and "Road" to 3.
- Customer_rating_category: Categorization of customer ratings into 'VeryLow', 'Low', 'Average', 'Standard', and 'High' for analysis.
- Prior_purchases_category: Categorization of prior purchases into 'Very Low', 'Low', 'Medium', and 'High' to assess the impact of customer engagement.
- Product_importance_category: Retention of original product importance categories for analysis.
- Gender_encoded: Encoding of gender into numerical values (0 for Female, 1 for Male).
- Interaction_CustomerRating_Discount: Interaction between customer rating and discount offered.
- Interaction_CustomerCalls_Rating: Interaction between customer care calls and customer rating.
- Shipping_speed: Calculation of shipping speed based on product weight and customer care calls.
- Total_interactions: Sum of customer care calls and prior purchases to measure customer engagement.
- Expected_delivery_time: Calculation of expected delivery time based on mode of shipment.
- Product_importance_avg_delivery: Calculation of average delivery performance for each product importance category.
- ShippingMode_avg_delivery: Calculation of average delivery performance for each shipping mode.
- Interaction_Weight_Discount: Interaction between product weight and discount offered.
- High_product_importance_and_high_rating: Identification of high product importance with customer rating 5.
- Weight_category: Categorization of product weights into 'Light', 'Medium', and 'Heavy'.
- Discount_category: Categorization of discount percentages into 'Low', 'Medium', and 'High'.
- Product_Value: Calculation of product value based on cost and weight.
- Customer_Loyalty: Categorization of customer prior purchases into 'New', 'Regular', and 'Frequent'.
- Customer_Satisfaction_Score: Calculation of average score of customer rating and delivery time.
- Delivery_Time_per_Weight: Calculation of delivery time per weight of the product.

![feature correlation](/image/Feature.png)

## Machine Learning model

- Naive Bayes: A probabilistic classification algorithm based on Bayes' theorem, commonly used for text classification and problems with high-dimensional feature spaces.
- Logistic Regression: A widely used classification algorithm for binary classification problems, modeling the probability of a binary outcome by fitting a logistic function to input features.
- K-Nearest Neighbours (KNN): A non-parametric classification algorithm that classifies data points based on the majority class of their K nearest neighbors. Suitable for small to medium-sized datasets.
- Support Vector Machine (SVM): A powerful classification algorithm that finds an optimal hyperplane to separate data points of different classes, effective in high-dimensional spaces.
- Decision Tree: A tree-based classification algorithm that recursively splits the dataset based on features to create a tree-like structure. Prone to overfitting on deep trees.
- Random Forest: An ensemble learning method that builds multiple decision trees and combines their predictions for more accurate results. Reduces overfitting.
- XGBOOST (Extreme Gradient Boosting): A gradient boosting algorithm known for its high performance and efficiency, which optimizes a loss function to improve prediction accuracy.

![ML]( /image/ml.png)

## Final result
1. Upload the csv
2. All data related operations will be performed including data preprocessing and ML.
3. Dashboard will be created with top features and model accuracy will be displayed
4. Questions will be answered by selecting the right model.
5. Recommendations will be displayed depending on the accuracy.

> Hope you watched [**Demo Video**](https://github.com/Ashleshk/Stride-Logistics-Analytics-Application/blob/main/DemoVideo.mp4)
> 
>
> Business then take a notch ahead my making insightful Dashboard as below
>
> [Tableau Dashboard](https://public.tableau.com/app/profile/ashlesh2213/viz/LogisticStrideAnalysis/LogisticsStrideAnalysis)

![Dashboard](/image/dashboard.png)
## Conclusion
A Stride Logistics Analysis project has been successfully completed, providing valuable insights to enhance Stride Logistics. By analyzing data, we answered crucial questions about delivery, customer satisfaction, and queries, offering practical recommendations that improved operations. 

Hope you Like. Drop **Star**!! 
Follow me on [Linkedin](https://www.linkedin.com/in/ashleshk/) 
