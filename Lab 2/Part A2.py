import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
"""
STATS LAB 6.1

Part 1. Collect the Data

We first begin by getting the dataset by webscraping, and convert it to a dataframe,
for easier handling. 
"""
# Exercise 1
print("-------------------Collect the Data-------------------")
url = "https://openstax.org/books/introductory-statistics/pages/c-data-sets"
df = pd.read_html(url, attrs={"data-id":"id5853402"})[0]
"""
With the help of the random library we take 6 random samples from each stratum,
and do that for column 2-7 and reconvert it to a dataframe.
"""
stratifiedTable = [random.sample(df.iloc[:,i].tolist(), 6) for i in range(2,8)]
stratifiedDF = pd.DataFrame(stratifiedTable).T
print(stratifiedDF)

#Exercise 2
"""
We first begin by stacking all the values in the dataframe so that pyplot
doesn't sort them by lap. Afterwards we group the data into six intervals and plot it.
"""
values = df.iloc[:,1:]
stackvalues = values.stack()
plt.hist(stackvalues,bins=6 ,rwidth=0.2)
plt.ylabel("Frequency")
plt.xlabel("Lap Time")
plt.title("Histogram of lap time with it's normal distribution curve")
#Exercise 3
"""
Here we take the mean of the stacked dataframe and round it to two decimal points.
"""
histavg = np.round(stackvalues.mean(), 2)
print("The mean value of the dataset is",histavg)
"""
Because this is stratified sampling we need to take into account the bias, 
therefore instead of taking the standard deviation of the whole dataframe we 
first need to take the sum of the mean variance for each stratum and then we
take the square root of that to get the standard deviation.
"""
varlist = values.var().tolist()
meanvar = sum(varlist)/len(varlist)
histstd = np.round(meanvar**0.5, 2)
print("The standard deviation of the dataset is",histstd)

#Exercise 4
"""
Description: The shape of the curve is a bell curve, the mu value is at the
average 129.74, by looking at it it looks a bit too much to the left, this is
because of the intervals. There is actually alot more values at 124 than at
136, which is why it is shifted more to the left.
"""
lsvalues = np.linspace(stackvalues.min(),stackvalues.max())
normdist = norm.pdf(lsvalues,histavg,histstd)
plt.plot(lsvalues, normdist*350)
"""
Part 2. Analyze the Distribution
"""
print()
print("-------------------Analyze the Distribution-------------------")
print("Since the mean is",histavg, "and the standard deviation is",histstd,
      "the theoretical distribution should be X ~ N(",histavg,",",histstd,").",
      "The histogram didn't help me at arriving to the approximate distribution.",
      "The histogram only shows how the data is distributed, not the specific values.")
"""
Part 3. Analyze the Distribution

We can directly see that the mean from the previous exercise and the median 
are not the same, which means thatour normal distribution is not 100% symetric.
"""
valuesnp = values.to_numpy()
empIQR1, empIQR3, empperc15, empperc85 = np.percentile(valuesnp, [25, 75, 15, 85])
print()
print("-------------------Describe the Data-------------------")
print("The IQR goes from", empIQR1,"to", empIQR3)
print("The IQR is", empIQR3-empIQR1)
print("The 15th percentile is", empperc15)
print("The 85th percentile is", empperc85)
print("The median is", np.median(valuesnp))
print("The empirical probability that a randomly chosen lap time is more than 130 seconds is",len(valuesnp[valuesnp>130])/len(stackvalues.tolist())*100,"%")
print("The 85th percentile shows the time you need to have a better time than 85% the other.")
"""
Part 4. Analyze the Distribution

Here we create a normal distribution from 100 000 random numbers with a mu and std. 
This solution will get us a different answer everytime.
We can also see that there is a big difference in the probability to chose a lap 
time greater than 130. This is due to the gerenation of the random numbers, np will
for example generate 130.01 and star counting from there while the empirical values,
start counting from 131 since only integers are in the dataset.
"""
probnormdist = np.random.normal(histavg, histstd, 100000)
probIQR1, probIQR3, probperc15, probperc85 = np.percentile(probnormdist, [25, 75, 15, 85])
print()
print("-------------------Theoretical Distribution-------------------")
print("The IQR goes from", np.round(probIQR1, 2),"to", np.round(probIQR3, 2))
print("The IQR is", np.round(probIQR3-probIQR1, 2))
print("The 15th percentile is", np.round(probperc15, 2))
print("The 85th percentile is", np.round(probperc85, 2))
print("The median is", round(np.median(probnormdist), 2))
print("The probability that a randomly chosen lap time is more than 130 seconds is",np.round(len(probnormdist[probnormdist>130])/len(probnormdist)*100, 2),"%")
print("The 85th percentile in the normal distribution shows what time you need to have a better time than 85% of the other times.")