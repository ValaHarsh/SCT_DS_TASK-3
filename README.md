# **Customer Purchase Prediction Using Decision Tree Classifier**

## **Project Overview**

This project is focused on building a **Decision Tree Classifier** to predict whether a customer will purchase a product or service based on their **demographic** and **behavioral** data. The dataset used is from the **UCI Machine Learning Repository**'s **Bank Marketing dataset**, which contains various features about customers and their interactions with the bank’s marketing campaign. The goal is to develop a machine learning model that can classify customers into two categories: those who will subscribe to a term deposit (yes) and those who will not (no).

---

## **Objective**

The primary objective of this project is to:
1. **Analyze the Bank Marketing dataset** to understand the factors influencing customer purchases.
2. **Clean and preprocess the data** to ensure it’s suitable for machine learning.
3. **Build and evaluate a Decision Tree Classifier** to predict customer behavior (purchase).
4. **Visualize** the data and model results for better interpretability.

---

## **Dataset Description**

The **Bank Marketing dataset** consists of demographic and behavioral features related to customers, including their age, job type, marital status, education level, and more. Below are some important features:
- **age**: Customer's age
- **job**: Type of job (e.g., admin, technician, services)
- **marital**: Marital status (e.g., single, married, divorced)
- **education**: Education level (e.g., primary, secondary, tertiary)
- **balance**: Average yearly balance in euros
- **housing**: Whether the customer has a housing loan
- **loan**: Whether the customer has a personal loan
- **contact**: Contact communication type (cellular, telephone)
- **previous**: Number of contacts performed before this campaign
- **pdays**: Number of days since the customer was last contacted during a previous campaign
- **y**: Target variable (whether the customer subscribed to the term deposit: "yes" or "no")

The dataset contains both **numerical** and **categorical** variables.

---

## **Methodology**

### **1. Data Cleaning and Preprocessing**
Data cleaning and preprocessing steps were necessary to handle the dataset's missing values, inconsistent data types, and categorical variables. The following actions were performed:
- **Handling missing values**: We imputed missing data for certain features or dropped columns that had too many missing values.
- **Encoding categorical variables**: Categorical variables (such as `job`, `marital`, `education`, etc.) were encoded using **LabelEncoder** to transform them into numerical representations suitable for machine learning.
- **Feature scaling**: Continuous variables like `age`, `balance`, and `duration` were standardized to ensure all features have the same scale.

### **2. Exploratory Data Analysis (EDA)**
Before building the model, we explored the data to understand:
- **Distribution of target variable**: We looked at the distribution of the target variable (`y`) to understand the class imbalance and distribution of customers who subscribed or not.
- **Feature correlations**: Visualized the relationships between features and the target variable using plots like **histograms**, **bar plots**, and **box plots**.
- **Feature importance**: By visualizing feature importance using the decision tree, we understood which variables had the most impact on predicting customer behavior.

### **3. Model Building**
The **Decision Tree Classifier** was chosen due to its interpretability and ability to handle both numerical and categorical data. The steps involved:
- Splitting the dataset into training (80%) and testing (20%) sets.
- Training the **Decision Tree model** using the training data.
- Fine-tuning hyperparameters such as `max_depth` and `min_samples_split` using **GridSearchCV** to enhance model performance.
  
### **4. Model Evaluation**
The model was evaluated using various metrics:
- **Accuracy**: Percentage of correct predictions.
- **Confusion Matrix**: To assess how well the model classified both classes (`yes` and `no`).
- **Precision, Recall, F1-Score**: To check the performance of the model, especially considering class imbalance.

---

## **Key Insights and Visualizations**
![Image](https://github.com/user-attachments/assets/19eb22a5-e03b-4209-97e4-11d48ab57e1d)
### **1. Customer Demographics**
Visualizations like **bar charts** helped understand the distribution of customer demographics and how they influence the likelihood of subscribing to a product. Insights like:
- Customers in **higher age groups** tend to have a higher subscription rate.
- **Married** individuals with a **tertiary education** are more likely to subscribe.

### **2. Feature Importance**
The **Decision Tree** model provides a clear visualization of which features are most important for predicting customer behavior. Features like `duration`, `previous`, and `age` were identified as the top predictors.

### **3. Model Performance**
- **Confusion Matrix**: We saw that the model performed well with a balanced prediction of both `yes` and `no` classes.
- **Accuracy**: The final model achieved an accuracy of around 90%, indicating strong predictive performance.
- **Hyperparameter Tuning**: Fine-tuning the model using GridSearchCV improved the performance slightly, confirming that hyperparameter optimization is crucial.

---

## **Conclusion**

The **Decision Tree Classifier** proved to be an effective model for predicting whether a customer would subscribe to a term deposit. By analyzing the demographic and behavioral data, we identified key factors that influence customer purchase decisions. The model achieved a high level of accuracy, and additional insights were gained through the visualizations.

### **Future Improvements**
- **Try Different Models**: We could explore other models like **Random Forest** or **Gradient Boosting** for potential improvement.
- **Feature Engineering**: We could explore new features, such as customer tenure, income, or product usage.
- **Handling Imbalanced Classes**: Implementing techniques like **SMOTE** or adjusting class weights could further enhance performance.

---

## **Technologies Used**
- Python: The programming language used for this project.
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
- Machine Learning: Decision Tree Classifier, GridSearchCV
- Data Visualization: Matplotlib, Seaborn

---

## **Acknowledgments**
- The Bank Marketing dataset from the UCI Machine Learning Repository.
- Scikit-learn documentation for machine learning algorithms.
- Matplotlib and Seaborn for data visualization tools.
