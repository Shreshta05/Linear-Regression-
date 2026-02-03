#import
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#create data
first_hour_likes = [120, 340, 89, 510, 230, 670, 45, 390, 150, 720,
                    280, 95, 440, 180, 560, 75, 320, 480, 210, 630,
                    110, 350, 85, 410, 260, 590, 140, 470, 55, 380]
    
total_views = [8500, 22000, 5200, 41000, 15000, 53000, 3200, 28000, 10500, 58000,
               18500, 6800, 32000, 12000, 44000, 4500, 21000, 36000, 14000, 49000,
               7800, 23500, 5000, 30000, 17000, 46000, 9500, 35000, 3800, 27000]

#convert and reshape it
x = np.array(first_hour_likes).reshape(-1, 1)
y = np.array(total_views)

#create a model
model = LinearRegression()
model.fit(x, y)

#predict - raj's question
new_likes = 750
predicted_views = model.predict([[new_likes]])
print(f"\n Raj's Question : If i got {new_likes} first hour likes, then how many views i will get?")
print(f"\n The predicted Total Views = {predicted_views[0]:.0f}")

#visualize
plt.figure(figsize = (10, 8))
plt.scatter(first_hour_likes, total_views, color = 'red', s = 100, label = 'Actual Data')
plt.plot(first_hour_likes, model.predict(x), color = 'green', linewidth = 2, label = 'Linear Regression Line')
plt.scatter(new_likes, predicted_views, color = 'blue', s = 200, marker = '*', label = 'Our Prediction')
plt.xlabel('First-Hour Likes', fontsize = 12, fontweight = 'bold')
plt.ylabel('Total views', fontsize = 12, fontweight = 'bold')
plt.title('Linear Regression : First-hour likes Vs Views', fontsize = 14, fontweight = 'bold')
plt.legend
plt.grid(True, alpha = 0.3)
plt.savefig('Instagram reel views predictor.png', dpi = 150, bbox_inches = 'tight')
plt.show()