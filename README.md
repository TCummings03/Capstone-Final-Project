## Capstone Final Project

The goal of this project is to create a machine learning model that predicts rent prices in the DC area.  This information would be useful not only to business (particularly those in real estate), but also to potential renters seeking an apartment in the area.  An analysis of this data will provide us insight into the most expensive/ least expensive localities and the main drivers of rent price.   

Instead of looking for a pristine data set that was ready to be split into test/training sets and used to predict rent prices, I chose to go out and get the data on my own.  This meant that I had to do some webscrapping in order to get the data for rent prices in the DC area. I chose to use the beautifulsoup library to accomplish this.

After choosing a well known rent listing website that had listings for the DC area, I had to break up the webscrapping into a few parts:

1. The site had 20 listings per page and around 700 pages, so the first thing I did was get all 700 pages' url's and store them in a          list.
2. Next, I used each of the above pages to get preliminary information on each listing by scrapping the number of bedrooms/bathrooms,        latitude/longittude, listing id, sqft, and price. I stored each of these items in their respective master list. In addition, I            scrapped each listing's respective url to get further information on the listing.
3. After getting the preliminary information and each listing's url, I went a level deeper and scrapped in indivdual listing page.            From here, I got a list of all the amenities each listing had and appended it to a master list.
 4. Lastly, I created a dataframe with all the scrapped elements

Webscrapping code: [Webscrapping](https://github.com/TCummings03/Capstone-Final-Project/blob/master/Capstone%20Webscraping.ipynb)

Although the scrapping was finished, my work did not stop there because I needed to unpack the amenities list and turn them into dummy variables so they could be used in a machine learning model.  First, I got the unique amenities from the amenities list and turned each of those amenities into an empty column.  Second, I looped over each row in the amenities column and added a 1 to the respective column where the amenity appeared. After creating the dummy variables for amenities, I checked to make sure there were no duplicates, which would provide noise to the model.  I found that there were in fact duplicates and instances like 'Washer' and 'washer' were treated as two separate variables. To remedy this, I turned all the amenities columns into lower case, grouped by column and summed them.  In addition, I had to make sure all of the numeric data was of data type numeric, so that it could be used in sci-kit learn.  I used pd.to_numeric to achieve this.  

Geocoding: Since I was able to scrape the latitude and longitude data for each listing, I decided to geocode this data and retreive the locality for each listing.  This would, perhaps, provide more insight into the spread of the listings on a physical map, and potential predictive power in a machine learning model. For this, I used the pygeocoder package which is a wrapper for Google's geocoding API. 

```#Geocoding to get neighborhood for each observation from longitude and latitude

hoods = []

G = Geocoder('YOUR_API_CODE_HERE')

for coor in tqdm(range(len(df))):
    add = G.reverse_geocode(df['lats'][coor], df['longs'][coor])
    if add.neighborhood == None:
        hoods.append(add.locality)
    else:
        hoods.append(add.neighborhood)
    time.sleep(0.5)
```

Outliers:

During my exploratory data analysis, I created scatter plots of each of the numeric columns(beds, baths, sqft) plotted against rent price to see if there were any outliers. From the scatter plots, I was able to detect the outliers and investigate those points. I could tell that they were outliers on the graph becuase they did not follow the general trend. However, upon further inspection I was able to confirm they were outliers because they had values such as 99999 for square footage or price. I removed these points.  In addition, I checked points below 100 sqft that had rent prices that did not seem plausible -- I also removed these points. In the below plots, we can see the progression of the plots after removing outliers.

![plot1](https://github.com/TCummings03/Capstone-Final-Project/blob/master/Images/Pic1.png)
--
![plot2](https://github.com/TCummings03/Capstone-Final-Project/blob/master/Images/Pic2.png)
--
![plot3](https://github.com/TCummings03/Capstone-Final-Project/blob/master/Images/Pic3.png)

Missing Values:

The main issues concerning missing values had to do with the amenities column and the sqft column.  About 10% of listings did not include any amenities at all, whereas nearly 30% of listings did not include square footage.  For the amenities columns, I removed all listings without any amenities, so the model could present a better sense of which(if any) amenities had an effect on rent price.  For the square footage column, I decided to run a simpler linear regression model with bedrooms and bathrooms to impute the missing values.  This was my process for imputing missing sqft values:

1. Create a data frame with sqft, bedrooms, bathrooms
2. Get all observations where sqft = NaN
3. Drop all NaN values from the smaller data frame
4. Fit a linear regression model on the smaller data frame
5. Take the obersvations where sqft = NaN and predict NaN values with linear model
6. Add these predictions to the original data frame (fill in missing sqft values)

After I had removed outliers, checked for duplicates, imputed missing values, and created all dummy variables, my data frame was ready for a machine learning model. Through this process, I was able to get hands on experience with messy data from the web and turn it into a manageable dataframe that was accessible. 

**Data Story**

[Visualizations](https://github.com/TCummings03/Capstone-Data-Story/blob/master/Capstone%20Visualizations.ipynb). Housing prices are some of the biggest expenses people incur on a monthly basis. Therefore, finding the best or fair price for rentals is crucial to making sure your money is well spent. Some important questions worth asking in order to gain this insight are:

1. What is the median/mean rental price and what does this distribution look like?
2. Where are the most expensive and cheapest rental neighborhoods? How do these values compare?
3. Does there seem to be a connection between location and price? Which neighborhoods are the most expensive?
4. How much more money does an extra bedroom cost? An extra bathroom? 
5. Prove Location, Location, Location


We will explore these questions below:


1. What does the rent price distribution look like?

![Distribution Prices](https://github.com/TCummings03/Capstone-Data-Story/blob/master/Images/download.png)

From the plot above, we can see that the distribution looks to be right skewed. In this plot, the mean and median are higher than the mode. In other words, there are a higher number of observations to the left of the mean. In addition, the mean is lifted higher as a result of some larger observations.  After checking these observations in the data wrangling portion of this project, we know that these larger values are not outliers. However, if this was initial exploratory data analysis, it would be wise to check the more extreme values to see if they are in fact valid points.  

The mean and median values are $2,366 and $2,100 respectively with a standard deviation of $1,148.51. In addition, the mean and median values based on sqft are $1.84 and $1.56 respectively with a standard deviation of $1.06.

2. Location! Location! Location!
![Link](https://github.com/TCummings03/Capstone-Data-Story/blob/master/Images/bokeh%20plot.png)

Looking at the distribution of rentals above by median locality price and colored by decile, we can see some interesting trends in the data. Perhaps, one of the most obvious trends is that the most expensive localities appear to be located in DC. Furthermore, as you move farther from DC (in all directions), rent prices appear to get cheaper with the darker points (lower deciles) located on the outskirts of DC. This makes sense intuitively as DC is an urban area with many attractions and the localities surrounding DC are suburbs. Let's take a closer look at the lower and upper deciles. Running the code for the graph above will provide an interactive plot that allows a user to search by decile and see what the median sqft price is along with which locality.  

Most/Least Expensive Rentals (Top and Bottom Decile)
![Upper Decile](https://github.com/TCummings03/Capstone-Data-Story/blob/master/Images/Top10.png)

From looking at the plot above, it looks like the median sqft median price is around $3 which is almost double the median value of $1.56. Furthermore, the localities with the highest median sqft price are Downton DC (Dupont) and Tysons, which have median sqft prices around $4 per sqft. It is also interesting to note that Tysons and Aurora Highlands appear to have the largest interquartile ranges with the edges of the boxes extending farther than other localities.  In addition, we should notice that Northwest Washington appears to have many listings outside of the whiskers of the boxplot, which suggests there may be a lot of outliers.

![Lower Decile](https://github.com/TCummings03/Capstone-Data-Story/blob/master/Images/Bottom10.png)

From the plot above, we can tell that there is a drastic different in terms of median sqft price. Whereas the top decile consisted of a median price around $3, the median price for the lower decile looks to be around $1 per square foot. However, we also know that from looking at our plot of deciles by locality, that these cheaper priced listings are located farther from DC proper.  Another interesting thing to note about this plot is that compared to the top decile plot, most of the localities appear to have listings that are relatively similar in price with smaller boxes and whiskers.  

3. Bedrooms and Bathrooms

Another important factor that affects the rental price is the number of bedrooms and bedrooms. Let's take a closer look at the marginal prices of these to see how much and extra bedroom or bathroom affects the rental price.  For this, we will look at the median prices:

![Bedrooms](https://github.com/TCummings03/Capstone-Data-Story/blob/master/Images/Bedrooms.png)

Price of an extra bedroom:

0.5 -1: $53

1 -2 : $250

2 -3 : $200

3 -4 : $595

4 -5 : $602

From the table above, we can see that studio apartments and one bedroom apartments are similarly priced, whereas rentals with multiple bedrooms seem to have a larger difference in price.  One would guess that the more bedrooms a rental has, the more expensive it is, and it seems like that is the general trend from the graph above. However, we can also notice that the difference between 1-2 bedrooms($250) is larger than the difference between 2-3 bedrooms ($200), which may suggest that 3 bedroom rentals are a better value when looking for multiple bedrooms.


![Bathrooms](https://github.com/TCummings03/Capstone-Data-Story/blob/master/Images/Bathrooms.png)

Price of an extra bathroom:

1-2: $450

2-3: $495

3-4: -$300

From the table above, it appears that having an extra bathroom seems to have a greater effect on rent price than an extra bedroom.  However, as we suspected from bedrooms, we'd expect that the more bathrooms an apartment has, the higher the price. This seems to be the case for most of the differences, however, an interesting deviation is the change from 3-4 bathrooms.  Here, 4 bathrooms is actually cheaper ($2295) than 3 bathrooms ($2595).  Perhaps there are other factors contributing to the difference in price here, and this would require further investigation.

4. The bigger, the more expensive?

![Rent Price vs SQFT](https://github.com/TCummings03/Capstone-Data-Story/blob/master/Images/Rent%20price%20vs%20SQFT.png)

As we have hypothesized earlier, the larger the apartment, the higher the price. We can see this fairly clearly in the plot above, which shows rent price vs. sqft.  In the plot above, we see a postive correlation between rent price and sqft implying the more sqft, the higher the price.  However, from looking at the deciles above, we have the added benefit of knowing where the value lies in terms of price per sqft broken down by decile.  

**Visualization Conclusions:**

From our visual exploration of the data, we have gained a greater insight into what effects rent price. First and foremost, we know that the number one rule in real estate holds true here: Location! Location! Location! The closer the rental is to DC, the higher the price.  In addition, we found out that the larger the rental (sqft), the higher the price.  Similarly, the more bedrooms/bathrooms a rental has, the higher the price and this makes sense because this generally means extra beds/baths lead to higher sqft, which in turn leads to a higher rental price. These intuitions will help lead us in the right direction when trying to build a model to predict rental price based on our features.

**Inferential Statistics**

For this portion of the project, I will use some of the inferential statistics techniques I've learned to gain greater insight into my cleaned data set.  The final goal for the capstone project will be to predict rent prices in the DC area by using a regression model.  One of the models we will try to use, is linear regression. Although normality of the dependent variables is not a requirement for linear regression, it would behoove us to check the distribution of rent prices to see if they are normally distributed.  

![Image](https://github.com/TCummings03/Springboard-Capstone-Inferential-Statistics/blob/master/Images/Regular%20distplot.png)

As noted in the Data Story, the distribution of rent prices seems to be right skewed. This may suggest that there are some extreme values pulling the mean of the distribution up and the median will be a better statistic for understanding the middle values.  Perhaps, taking the log of the rent prices will provide us with a normally distributed plot.

![Image](https://github.com/TCummings03/Springboard-Capstone-Inferential-Statistics/blob/master/Images/Logfit%20displot.png)

As we can see from the plot above, the distribution appears to look more normal after the log transformation. Included on the plot, in black, is a theoretical normal distribution for the values we have.  The distplot appears to be pretty close to this normal projection.  However, for interpretation purposes, it is important to note that if we were to create a regression model with rent price(y) and indepdent variables(x), the betas of the independent variables would be the percentage increase in y as a result of a one unit increase in x.  

What are the most important independent variables?

In order to answer this question, we will create a correlation matrix with all of the independent variables and dependent variable to see which 9 have the highest correlation with rent price in order to gain greater insight into what effects the price.

**Highest Positive Correlation**

![Image](https://github.com/TCummings03/Springboard-Capstone-Inferential-Statistics/blob/master/Images/Positive%20Corr.png)


**Highest Negative Correlation**

![Image](https://github.com/TCummings03/Springboard-Capstone-Inferential-Statistics/blob/master/Images/Negative%20Corr.png)

As we can see from the tables above, the top three most influential features according to the correlation matrix are sqft, bathrooms, and bedrooms (in that order).  The next closest feature has half the correlation coefficient of bedrooms ('Cooktop').  It's also interesing to note that the negatively correlated features have fairly weak correlations with the highest being around -0.12.  Perhaps even more interesting, the classic trope, Location! Location! Location! seems not to be as important as the top three features. Northwest Washington is the highest correlated locality with price and it's correlation coefficient is only 0.203816.  Because of this, we will take a closer look into the top three correlated with price. Lastly, it is always important to note whenever dealing with correlations that correlation != causation. 

**Top Three Correlations**

![Image](https://github.com/TCummings03/Springboard-Capstone-Inferential-Statistics/blob/master/Images/Rent%20vs%20SQFT.png)

![Image](https://github.com/TCummings03/Springboard-Capstone-Inferential-Statistics/blob/master/Images/Rent%20vs%20Baths.png)

![Image](https://github.com/TCummings03/Springboard-Capstone-Inferential-Statistics/blob/master/Images/Rent%20vs%20Beds.png)

From the joint plots above, we can see a couple of different things. 1) A histogram of each variable 2) the pearsonr & p value 3) regression line.  All three have positive correlations with price, which suggests the more sqft, beds, or baths, the higher the rent price. This makes sense because the larger the apartment, the higher the price. What's interesting to note, however, about the sqft distribution is that it appears to be suffering from heteroskedasticity, which occurs when the variability of a variable is unequal acorss the range of values of a second variable that predicts it.  Due to the hetereoskedasticity, it may be wise to also take a log transformation of sqft when running our regression model. The distribution of bedrooms and bathrooms seems to be right skewed, which may suggest that they too could benefit from log transformations.  Although this is not a requirement for linear regression, it may help improve our model and make interpretation of coefficients easier. As far as the central limit is concerned, we can be comfortable with satisfying the benchmark of more than 30 observations since our dataset consists of nearly 12,000 listings. 

**Base Case**

Figuring out the best loss function to score our model is crucial. Whether you're using RMSE, MSE, r^2, etc. it is important to define your scoring fucntion.  Another critical inferential statistic skill that is necessary for analyze our results is creating a "baseline case." This baseline case will help serve as the backdrop against which we can compare the results of our model.  We will use this baseline case in conjunction with our scoring function of Root Mean Squared Error to evaluate our model.  The base case is found by taking the median price/ median sqft and multiplying it by each respective listing's sqft. This "base case" is a fairly crude way of predicting rent price. However, this is useful as a bench mark to see if using a model will actually be useful in trying to predict rent price.  At a minimum, the model should do better than the RMSE from the base case (1101.4502853). Here is the code for the base case:

```#Get a baseline case to compare model against:

median_price = df1.rent_price.median()

median_sqft = df1.sqft.median()

median_p_sqft = median_price/median_sqft

baselinep = [median_p_sqft * x for x in df1.sqft]

err = df1.sqft - baselinep

sq_err = err ** 2

mean_sq_err = np.mean(sq_err)

root_mse = np.sqrt(mean_sq_err)

print('Median Rent Price:', df1.rent_price.median())
print('Mean Rent Price:', df1.rent_price.mean())
print('RMSE Base Case:', root_mse)

Median Rent Price: 2100.0
Mean Rent Price: 2365.5429759257468
RMSE Base Case: 1101.4502853
```

**Machine Learning**

I have included a link to my analysis/annotated code of machine learning models using sci-kit learn [here](https://github.com/TCummings03/Capstone-Final-Project/blob/master/Capstone%20Machine%20Learning.ipynb).

### Conclusions & Takeaways ###

1. Location! Location! Location!

The number one rule of real estate (Location! Location! Location!) holds in this case. After running all of the machine learning models and settling on RidgeCV since it had the best(lowest) RMSE, I checked the coefficients to see which were having the biggest impact on rent price. As we can see from the tables below, the top 10 and bottom 10 values are all localities, which suggests that Location has the biggest impact on rental price. 

Top 10 Coefficients:

![Top10](https://github.com/TCummings03/Capstone-Final-Project/blob/master/Images/Top10coeff.png)

Bottom 10 Coefficients:

![Bottom10](https://github.com/TCummings03/Capstone-Final-Project/blob/master/Images/Bottom10coeff.png)


2. How many bedrooms and bathrooms?

For individuals looking to live by themselves, they have two choices: studio apartment or a one bedroom apartment. However, from previous analysis, we know that the median extra cost of a bedroom (moving from a studio to a one bedroom) is only $53 compared to going from a 1 bedroom to a 2 bedroom which is $250. For these individuals, it may be worth it to pay slightly more for the extra bedroom and space.

In addition, the marginal price of a "half bathroom" is relatively small, so rental listings that include huge differences for adding the half bathroom should give us pause.  

3. What about just DC?

DC is the metropolitan center around which the rest of the rental listings lie.  But what about the prices in DC? Where is the best value hidden in America's capital city? Here is the list of median sqft price by each section of DC:

Downtown DC : $3.95

Northwest DC: $3.07

Southwest DC: $2.29

Southeast DC: $1.65

Not surprisingly, the most expensive area in DC is downtown DC which commands a median sqft rental price of $3.95, which is not only the most expensive in DC, but the most expensive in the entire DC area! However, for those people looking to move to the DC area, and particularly want to settle down in the District, the best value seems to lie in southeast DC, which commands a median sqft price of $1.65.  Although these apartments are farther away from downtown DC and all of the attractions, it may be worth the more than 100% premium one would pay to live downtown.
 

A few notes:

1. One of the biggest issues that I ran into when creating a webscrapping tool was the presence of outliers.  This would greatly impact the accuracy of my model and it was necessary to take precaution when creating a model. Going foward, it would help to have an outlier detection model that would return a probability for whether or not a listing is a 1. a real listing with corretctly inputed data 2. an outlier

2. Another way to improve this model and provide more insight would be to find points of interest and include the distance from the listing to the point of interest as a feature. This would require some GIS knowledge.

3. A lot of the amenities did not seem to provide material predictive power on the model, and I believe this is because the sheer number (1400+) created too much noise combined with each amentiies being subject to user input errors. A text analysis of the "soup" of amenities could provide greater insight into the actual distribution of amenities.
