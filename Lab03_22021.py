# 1 and 2
import pandas as pd
import numpy as np

df = pd.read_excel("C:\Users\injet\Desktop\Lab Session1 Data.xlsx")
# Example: Extracting specific columns into matrices
matrix1 = df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].values  
matrix2 = df[['Payment (Rs)']].values
print(matrix1)
print(matrix2)
num_rows, num_columns = matrix1.shape
print("Dimensionality of the vector space:", num_columns)
print("The number of vectors that exist in the vector space:", num_rows)
np_matrix = df.to_numpy()
rank = np.linalg.matrix_rank(matrix1)
print("Rank of the matrix:", rank)
pseudo_inv=np.linalg.pinv(matrix1)
X=pseudo_inv@matrix2
print("The individual cost of a candy is: ",round(X[0][0]))
print("The individual cost of a mango is: ",round(X[1][0]))
print("The individual cost of a milk packet is: ",round(X[2][0]))



#3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def classifier(df):
    features = ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]
    X = df[features]
    y = df['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    df['Predicted Category'] = classifier.predict(X)
    return df


# Load the dataset into a pandas DataFrame
df = pd.read_excel("C:\Users\injet\Desktop\Lab Session1 Data.xlsx")

# Create the 'Category' column based on the payment amount
df['Category'] = df['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')

# Run the classifier function
df = classifier(df)

# Print the relevant columns
print(df[['Customer', 'Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)', 'Category', 'Predicted Category']])


# 4
import pandas as pd
import statistics
import matplotlib.pyplot as plt

excel_file_path = "C:\Users\injet\Desktop\Lab Session1 Data.xlsx"
df = pd.read_excel(excel_file_path, sheet_name='IRCTC Stock Price')

price_mean = statistics.mean(df['Price'])
price_variance = statistics.variance(df['Price'])
print(f"Mean of Price: {price_mean}\n")
print(f"Variance of Price: {price_variance}\n")

wednesday_data = df[df['Day'] == 'Wed']
wednesday_mean = statistics.mean(wednesday_data['Price'])
print(f"Population Mean of Price: {price_mean}\n")
print(f"Sample Mean of Price on Wednesdays: {wednesday_mean}\n")


april_data = df[df['Month'] == 'Apr']
april_mean = statistics.mean(april_data['Price'])
print(f"Population Mean of Price: {price_mean}\n")
print(f"Sample Mean of Price in April: {april_mean}\n")


loss_probability = len(df[df['Chg%'] < 0]) / len(df)
print(f"Probability of making a loss: {loss_probability}\n")
wednesday_profit_probability = len(wednesday_data[wednesday_data['Chg%'] > 0]) / len(wednesday_data)
print(f"Probability of making a profit on Wednesday: {wednesday_profit_probability}\n")
conditional_profit_probability = wednesday_profit_probability / loss_probability
print(f"Conditional Probability of making profit, given today is Wednesday: {conditional_profit_probability}\n")
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
