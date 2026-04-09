# CMPE188-FlightDelayProject


**Project title:** Flight Delay Prediction via Train XGBoost model and GridSearchCV.
<br><br>


 **Team member names:** 
 
 Harinandan Kotamsetti 
 
 Angelina Ryabechenkova
 					
 Aisha Syed			
 <br><br>
 
 **Problem statement:**
 
Flight delays are a frequent and frustrating issue in air travel, often resulting from a combination 
 
 of operational, environmental, and logistical factors. These delays can disrupt passenger schedules, 
 
 increase costs, and reduce overall travel efficiency. Given the complexity and variability of these 
 
 factors, predicting flight delays in advance remains a challenging but valuable task.

In this project, we aim to develop a machine learning-based system capable of predicting the likelihood
	
of flight delays using a large dataset of over 500,000 domestic U.S. flights. The dataset includes key 

features such as airline, flight number, origin and destination airports, day of the week, departure time, 

and flight duration. By analyzing patterns within this data, the model will learn relationships between 

these variables and delay outcomes.

The system will be deployed as an interactive web application, where users can input specific flight 
	
details and receive a real-time prediction of whether a flight is likely to be delayed. In addition to 

predictions, the application will provide insights into contributing factors such as route congestion, 

airline performance trends, and time-based patterns.

The primary goal of this project is to assist travelers in making more informed decisions by anticipating
	
potential delays before they occur. By leveraging data-driven predictions, the system aims to improve travel 

planning, reduce uncertainty, and enhance the overall passenger experience. Furthermore, this project 

demonstrates how machine learning techniques can be applied to real-world transportation problems, 

highlighting the practical value of predictive analytics in improving operational efficiency and customer 

satisfaction.
<br><br>

**Dataset or data source:**  [Kaggle dataset](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay?phase=FinishSSORegistration&returnUrl=/datasets/jimschacko/airlines-dataset-to-predict-a-delay/versions/1?resource=download&SSORegistrationToken=CfDJ8OuP2e3tnwRHgoiuJLZ8zbysdCnRbnsgcridZRXpr-NuS8q6RErNazcFQSnT4nwgx8ls7nkHNv4s2bUj7M5a7MyJZ-Lljvdj3lDv66ynv-foNATtycZCW93r5RH2nR9uS3hShogxMpOAAwJ2pkGnNukIBZmB-14kmOeb2NJW_5mpXb3v-n0fDfHqIHfmsP9FBzC0cUY2J_VMvBAOx9ZAfqeLwZe7EP3fR3idsKFThZhz29Bfgf3skGRJtnd8h4yE0NsclMIOcgQ1IFDO0JWqypCpHj4xH0lfcU75FfpbmDgES-HX7IO2xhdjAdV1KD_QIOzNI-YGFYMf2EaD12q6I3EpHQPk8Wt2CRTEcTn8&DisplayName=Angelina%20Ryabechenkova)
 
 <br> 
 
**Planned model/system approach:**

Raw data --> Split Columns (ColumnTransformer) --> Encode Categories + Scale Numbers -->

Select Best Features --> Train XGBoost Model --> GridSearchCV refinement.

We split data into columns, then convert text to numbers with data encoding (using 1-hot), since ML cannot read 

text,  and needs numbers. Next, we select best features, as features like flight number have no affect on the delay

 probability. We use XGBoost Model, which is a tree-based ML model to predict delays based on the selected features.
 
 Lastly, we use GridSearchCV to try many combinations and find the best option to make predictions more effectively. 

 <br>
 
**Current implementation progress:** We had a chat about the project.