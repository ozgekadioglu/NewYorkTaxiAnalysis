# NewYorkTaxiAnalysis

## Problem Definition
Predicting the average money that a taxi driver in a given region of NY-city makes per day and hour.

The problem was approached as supervised-regression problem as the actual value of the target feature we are trying to predict is given and what is predicted is a continuous variable.

## About the Original Dataset
Metadata for the yellow taxies in New York-city was downloaded only for January 2019. More datasets can be found in the website of [Taxi&Limousine Commission (TLC)](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).  The list of features that is involved in the original data dictionary are: [‘tpep_pickup_datetime’, ‘tpep_dropoff_datetime’, ‘passenger_count’,‘trip_distance’, ‘RatecodeID’, ‘PULocationID’, ‘DOLocationID’, ‘payment_type’, ‘total_amount’]. More information regarding these features can be found in [that](https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf) link.

## Data Cleaning
Upon exploring the dataset column by column, it was seen that there are some outliers and many non-sense values for which the corresponding data points need to be removed. 

1- Negative ‘total_amount ‘ values. (7131 data points)

2- Too high ‘total_amount ‘ values. The upper limit was decided to be 200 dollars. For this decision I looked at the number of datapoint that needs to be removed for the cases of 100 (10832 points), 200 (1166 points), and 300 dollars (307 points).  We chose 200 dollars since it is not a big loss of points considering the size of the dataset (7667792 data points).

![Negative and zero values](/figures/figure2)


For a closer look on the negative and too high values, below only the data points where ‘total_amount’ is smaller than 1000 dollars is plotted.

![Too high values](/figures/figure1)

taxi_data_cleaned = taxi_data[(taxi_data[‘total_amount’]<200) & (taxi_data[‘total_amount’]>0)]


## Feature Engineering
/Data Related Features:/  We added a column showing which day of the week it was when the transaction happened. This helped us to identify the days as week and weekend days. By using USFederalHolidayCalendar, we were able to find out if it was a significant day for the transactions such as a public holiday. 

/Location Related Features:/  The LocationID of the places were already listed with the corresponding names on the TLC website. Another column was added for the Boroughs.

/Weather Related Features:/  The dataset gives information on the weather of NY-city on 2019. The features shared in the original dataset are temperature, humidity, wind speed, cloud cover, and amount of precipitation that were kept every three hours. Therefore, upon the merge of weather data and the transaction dataset, missing values appeared for the hours for which there is no weather data. These missing values were filled by doing interpolation and backward filling.

The columns in the final dataset: [‘PULocationID’, ‘transactionDate’, ’transactionMonth’, ‘transactionDay’, ’transactionHour’, ‘total_amount’, ’trip_distance’, ‘transaction_amount’, ‘transaction_week_day’, ‘weekend’, ‘holiday’, ’Borough’, ‘temperature’, ‘humidity’, ‘wind speed’, ‘cloud cover’,’ amount of precipitation’]-.

## Comparison of Different Algorithms 
Decision Trees, Random Forest and Gradient Boosting were implemented and compared. The benchmark model is a Decision Tree. For the benchmark, only the columns in the original dataset were used. The new columns added during the feature engineering were used for the rest of the models.

The performance results of the 3 algorithms are:

|                     | MAE      | RMSE      | R2       |
|---------------------|----------|-----------|----------|
| Benchmark           | 9.757872 | 14.703044 | 0.215522 |
| Decision Tree       | 8.468280 | 13.868055 | 0.302093 |
| Random Forest       | 7.345874 | 13.014364 | 0.385372 |
| Gradient Boosting   | 8.336377 | 13.221230 | 0.365678 |

The model fitted using Random Forest algorithm was decided to be worked on for the tuning. Correct hyper-parameters are tuned to improve the model performance. The best parameter values for the Random Forest Algorithms  were found to be:  ’n_estimators’: 400, ‘min_samples_split’: 20, ‘min_samples_leaf’: 4, ‘max_features’: ‘auto’, ‘max_depth’: 300, ’bootstrap’: True. After tuning the hyper-parameters, evaluation metrics of the Random Forest model became:

|                     | MAE      | RMSE      | R2       |
|---------------------|----------|-----------|----------|
| Benchmark           | 9.757872 | 14.703044 | 0.215522 |
| Decision Tree       | 8.468280 | 13.868055 | 0.302093 |
| Random Forest       | 7.345874 | 13.014364 | 0.385372 |
| Gradient Boosting   | 8.336377 | 13.221230 | 0.365678 |
| Tuned Random Forest | 7.156280 | 12.556617 | 0.427848 |

Below is the true vs. predicted value plot for the tuned random forest model. The true values are resented by the x-axis and the predicted ones by the y-axis.

![True vs. predicted value plot for the tuned random forest model.](/figures/figure3)


## Outlook and Suggestions

Once the hyper-parameters were tuned, the performance of the Random Forest improved. Therefore, there still might be possibility to improve the performance.  Here are some suggestions that were not tried in the code:

The dataset had information about the LocationID and the Boroughs as we added later to the original dataset.  Below you can see the results of pd.value_counts() for the column of ‘Borough’ in the dataset, which shows how many times each unique location appears in the dataset. 
|                             |       |   |   |   |
|-----------------------------|-------|---|---|---|
| Manhattan                   | 45309 |   |   |   |
| Brooklyn                    | 23632 |   |   |   |
| Queens                      | 21971 |   |   |   |
| Bronx                       | 9584  |   |   |   |
| Unknown                     | 1453  |   |   |   |
| Staten Island               | 302   |   |   |   |
| EWR                         | 270   |   |   |   |
| Name: Borough, dtype: int64 |       |   |   |   |


In the analysis, we did not limit the regions while training the model. There are some locations which do not have as much transactions as others do. If we were to avoid these locations, of course depending on the problem and the purpose, the performance could be improved. If the purpose is to sample all of NYC then we should definitely keep those data points.  But if the goal is simply increasing the performance of the model, then including only the borough with the highest amount of transactions will a good idea since the boroughs with fewer data points are more likely to bring mistakes into the model. However, such a decision should be made carefully.

Using random_grid(), the new hyper-parameters that lead better performance were searched and found. However, after finding the best parameters from this search, we did not continue searching for a better performance. Perhaps by comparing two models with comparable parameters but with significantly different performances, one can get insights towards a new set of random_grid and a model with better performance might be attained.

There are 1270 data points with total_amount=0 and trip_distance=0 which were removed during the data_cleaning since it makes sense that no trip distance means no payment. However, there are 51696 data points with trip_distance=0 while the total_amount>0. 51696 sounds like an important amount of data points to remove so I preferred not removing them. However, in real life, I would demand an explanation about the trips for which a payment happened even though the trip_distance is 0. The explanation could be the way trip_distance is calculated or a technical problem that occurred in the calculation system during the trip. Either way, for these data points, I would ask for an explanation to the data owner.

