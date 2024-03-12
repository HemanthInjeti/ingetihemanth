# 1 and 2
import pandas as pd
import numpy as np

# Read data from an Excel file into a pandas DataFrame
df = pd.read_excel("C:\Users\injet\Desktop\Lab Session1 Data.xlsx")

# Example: Extracting specific columns into matrices
# Extracting columns 'Candies (#)', 'Mangoes (Kg)', and 'Milk Packets (#)' into matrix1
matrix1 = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values  
# Extracting column 'Payment (Rs)' into matrix2
matrix2 = df[['Payment (Rs)']].values

# Displaying matrices
print(matrix1)
print(matrix2)

# Determining the dimensionality of the vector space (number of features)
num_rows, num_columns = matrix1.shape
print("Dimensionality of the vector space:", num_columns)

# Determining the number of vectors in the vector space (number of data points)
print("The number of vectors that exist in the vector space:", num_rows)

# Converting DataFrame to numpy array
np_matrix = df.to_numpy()

# Calculating the rank of matrix1
rank = np.linalg.matrix_rank(matrix1)
print("Rank of the matrix:", rank)

# Calculating the pseudo-inverse of matrix1
pseudo_inv = np.linalg.pinv(matrix1)

# Solving the linear system using the pseudo-inverse to find individual costs
X = pseudo_inv @ matrix2
print("The individual cost of a candy is:", round(X[0][0]))
print("The individual cost of a mango is:", round(X[1][0]))
print("The individual cost of a milk packet is:", round(X[2][0]))




#3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def classifier(df):
    """
    Function to classify customers based on their purchase history.
    
    Parameters:
    - df: Pandas DataFrame containing customer data
    
    Returns:
    - Modified DataFrame with predicted categories added
    """
    # Define features and target variable
    features = ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]
    X = df[features]  # Features
    y = df['Category']  # Target variable
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Initialize and train the logistic regression classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    
    # Predict categories for all data points
    df['Predicted Category'] = classifier.predict(X)
    
    return df


# Load the dataset into a pandas DataFrame
df = pd.read_excel("C:\Users\injet\Desktop\Lab Session1 Data.xlsx")

# Create the 'Category' column based on the payment amount
df['Category'] = df['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')

# Run the classifier function
df = classifier(df)

# Print the relevant columns
print(df[['Customer', 'Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)', 'Category', 'Predicted Category']])


# 4
import pandas as pd
import statistics
import matplotlib.pyplot as plt

# Path to the Excel file
excel_file_path = "C:\Users\injet\Desktop\Lab Session1 Data.xlsx"

# Read data from the Excel file into a pandas DataFrame
df = pd.read_excel(excel_file_path, sheet_name='IRCTC Stock Price')

# Calculate mean and variance of the 'Price' column
price_mean = statistics.mean(df['Price'])
price_variance = statistics.variance(df['Price'])
print(f"Mean of Price: {price_mean}\n")
print(f"Variance of Price: {price_variance}\n")

# Filter data for Wednesdays and calculate mean price
wednesday_data = df[df['Day'] == 'Wed']
wednesday_mean = statistics.mean(wednesday_data['Price'])
print(f"Population Mean of Price: {price_mean}\n")  # Note: This prints the overall population mean again
print(f"Sample Mean of Price on Wednesdays: {wednesday_mean}\n")

# Filter data for April and calculate mean price
april_data = df[df['Month'] == 'Apr']
april_mean = statistics.mean(april_data['Price'])
print(f"Population Mean of Price: {price_mean}\n")  # Note: This prints the overall population mean again
print(f"Sample Mean of Price in April: {april_mean}\n")

# Calculate probability of making a loss
loss_probability = len(df[df['Chg%'] < 0]) / len(df)
print(f"Probability of making a loss: {loss_probability}\n")

# Calculate probability of making a profit on Wednesdays
wednesday_profit_probability = len(wednesday_data[wednesday_data['Chg%'] > 0]) / len(wednesday_data)
print(f"Probability of making a profit on Wednesday: {wednesday_profit_probability}\n")

# Calculate conditional probability of making profit given it's Wednesday
conditional_profit_probability = wednesday_profit_probability / loss_probability
print(f"Conditional Probability of making profit, given today is Wednesday: {conditional_profit_probability}\n")

# Scatter plot of Chg% against the day of the week
day=['Mon','Tue','Wed','Thu','Fri']
day1=[]
chg1=[]
for i in day:
    for j in range(2,len(df['Day'])):
        if i==df.loc[j,'Day']:
            day1.append(i)
            chg1.append(df.loc[j,'Chg%'])
plt.scatter(day1, chg1)
plt.xlabel('Day of the Week')
plt.ylabel('Chg%')
plt.title('Scatter plot of Chg% against the day of the week')
plt.show()

