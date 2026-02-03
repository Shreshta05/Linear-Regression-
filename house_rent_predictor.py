#goal :- to build a rent prediction model
#using Linear regression model

#step 1 : import all libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#step 2 : data( house size in sqft , rent )

house_size = [500, 750, 900, 1050, 1150, 1200, 1500, 1800, 2300, 2600]
rent = [8000, 10000, 12000, 15000, 17000, 18000, 23000, 27000, 29000, 33000]

#step 3 : convert data into numpy array to reshape it

x = np.array(house_size).reshape(-1, 1)   #reshape is to arrange the data into vertical list
y = np.array(rent)

#step 4 : create a model and train that

model = LinearRegression()
model.fit(x, y)    #training the model on the x and y values

#step  : predict

new_house_size = 1700
predicted_rent = model.predict([[new_house_size]])   #i want an array so []
print(f"House size is {new_house_size} sqft")
print(f"Predicted rent is {predicted_rent[0]:.0f}")   #0 index value from array, if there is no array then no need of index, .0f is to remove decimal points

#step 6 : to look at the data and understand / visualize

plt.figure(figsize=(10, 6))  #size of the figure
#actual data
plt.scatter(house_size, rent, color='red', s=100, label = 'Actual Data' )  #to put the data points at the intersection of x and y axis on graph, color ansd size of dots and what the dots mean is actual data points and it doesnt give line it gives points
plt.plot(house_size, model.predict(x), color = 'green', linewidth = 2, label = 'Predicted values')
#predicted data
plt.scatter(new_house_size, predicted_rent, color = 'blue', s = 200, marker = '*', label = 'Our predictions')

plt.xlabel('House sizes in sq.ft', fontsize = 12)
plt.ylabel('Rents in rupees', fontsize = 12)
plt.title('House Rent Predictions', fontsize = 16)
#legend = shows meaning of colour
plt.legend
plt.grid(True, alpha = 0.3) # 0.3 is the size of the grid line a thin line
#to save the fig, dpi = dots per inch - pixel fig
plt.savefig('house_rent_predictions.png', dpi = 150, bbox_inches = 'tight' )

plt.show()

print("\n Graph saved as 'house_rent_predictions.png'")