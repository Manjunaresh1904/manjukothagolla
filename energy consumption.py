#1. Data creation and loading 

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

# Constants
num_zones = 5
days_in_year = 365
start_date = datetime(2023, 1, 1)

# Generate the synthetic dataset
def generate_data():
    data = []
    
    for day in range(days_in_year):
        date = start_date + timedelta(days=day)
        zone_id = random.choice(range(1, num_zones + 1))
        avg_temp = np.random.normal(loc=20, scale=5)  # Avg Temp: ~20°C with some variation
        humidity = np.random.uniform(30, 80)  # Humidity: Between 30% and 80%
        special_event = random.choice([0, 1])  # Randomly choose if there was a special event
        energy_consumption = np.random.normal(loc=1000, scale=200) + (avg_temp - 20) * 20  # Energy usage affected by temperature
        
        data.append([date, zone_id, avg_temp, humidity, special_event, energy_consumption])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['Date', 'ZoneID', 'AvgTemperature', 'Humidity', 'SpecialEvent', 'EnergyConsumption'])
    
    return df

# Generate synthetic data
df = generate_data()

# Show the first few rows
df.head()
#2. Data Analysis 
# Convert Date to datetime type for easier analysis
df['Date'] = pd.to_datetime(df['Date'])

# Calculate average energy consumption per month and per zone
df['Month'] = df['Date'].dt.month
monthly_avg = df.groupby(['Month', 'ZoneID'])['EnergyConsumption'].mean().reset_index()

# Calculate correlation between features
correlation_matrix = df[['AvgTemperature', 'Humidity', 'SpecialEvent', 'EnergyConsumption']].corr()

# Display results
print("Monthly Average Energy Consumption per Zone:")
print(monthly_avg)

print("\nCorrelation Matrix:")
print(correlation_matrix)
#3. Visualization 
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Line chart of monthly energy usage trends for all zones
plt.figure(figsize=(10, 6))
sns.lineplot(data=monthly_avg, x='Month', y='EnergyConsumption', hue='ZoneID', marker='o')
plt.title('Monthly Average Energy Consumption per Zone')
plt.xlabel('Month')
plt.ylabel('Energy Consumption (kWh)')
plt.legend(title='Zone ID')
plt.show()

# 2. Heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Features')
plt.show()

# 3. Bar chart comparing average usage on event vs non-event days
event_avg = df.groupby('SpecialEvent')['EnergyConsumption'].mean().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(data=event_avg, x='SpecialEvent', y='EnergyConsumption')
plt.title('Average Energy Consumption on Event vs Non-Event Days')
plt.xlabel('Special Event (0=No, 1=Yes)')
plt.ylabel('Average Energy Consumption (kWh)')
plt.xticks([0, 1], ['No Event', 'Event'])
plt.show()
#4. Prediction model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Prepare data for prediction (use previous day's data to predict next day's consumption)
df['PrevDayEnergyConsumption'] = df.groupby('ZoneID')['EnergyConsumption'].shift(1)

# Drop rows with missing 'PrevDayEnergyConsumption'
df.dropna(subset=['PrevDayEnergyConsumption'], inplace=True)

# Define features (X) and target (y)
X = df[['AvgTemperature', 'Humidity', 'SpecialEvent', 'PrevDayEnergyConsumption']]
y = df['EnergyConsumption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")