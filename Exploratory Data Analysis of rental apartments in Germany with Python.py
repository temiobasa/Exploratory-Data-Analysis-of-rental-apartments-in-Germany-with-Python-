#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis of rental apartments in Germany with Python 

# ### Client - Alaseju Property Development Company 
# ### Junior Data Scientist - Temitope Obasa

# ## Introduction 
# 
# #####  Please note that Alaseju Property Development company is a fictional company created solely for this project. Any resemblance to a real company is merely coincidental. 
# 
# #### This Exploratory Data Analysis project was carried out on behalf of the client, Alaseju Property Development Company. The Alaseju Property Development company, established in 2015, is a real estate development company that focuses on the residential property niche. The client seeks to enter the residential real estate market in the Brandenburg region. To provide a clearer understanding of the market, the client requested for a market research on the property market in the Brandenburg region. 

# #### The dataset used for this project was obtained from Kaggle. The dataset contains data on residential real estate offers that were available between 2018 to 2020 in Germany.
# 
# #### The data was scraped from Immoscout24 (www.immobilienscount24.de) by CorrieBar https://www.kaggle.com/corrieaar a Data Scientist. It contains details such as apartment type, base rent, total rent, facilities, type of heating, data, city, locality, state, service charge and apartment size amongst others. The date column contains the time the data was scraped from the Immoscout24.  

# ## Business Problem 
# 
# #### The Business Development Manager is convinced that developing in the Brandenburg State, Germany will be a profitable investment for the company in the coming years. He requested insights on the real estate market in Brandenburg State. In response, the Data Analytics Team embarked on an exploratory data analysis on the rental prices within the region. They hoped to answer the following;
# 
# #### - The market trend in Brandenburg
# #### - The market trend total rent within the Brandenburg region
# #### - The market trend in the city with the most expensive base rent
# #### - The price per metre square in Potsdam
# #### - Apartment size in Potsdam
# #### - Common Apartment Facilities

# ## Exploratory Data Analysis 
# 
# #### Exploratory Data Analysis is an approach to analyzing datasets with the goal of summarizing the main characteristics of the dataset, often using statistical graphics and other data visualization methods. 

# ## Importing Libraries

# In[1]:


# Import essential libraries for Data Analysis

import pandas as pd
import numpy as np

# Import essential library for splitting datetime
import datetime as dt

# Importing essential library for downloading the dataset from Kaggle
# import opendatasets as od

# Import essential libraries for Exploratory Data Analysis and Visualization
import matplotlib as ploty
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode
from scipy.stats import iqr
import plotly.express as px

# Ignore Warnings

import warnings
warnings.filterwarnings("ignore")


# ## The Dataset

# ### Loading the dataset

# In[2]:


# This loads the dataset and prints the first five rows in the dataframe

df = pd.read_csv("/Users/temi/Datasets/realestate/immo_data.csv")
df.head(5)


# #### It is also helpful to view the last five rows of the dataframe.

# In[3]:


df.tail()


# ### Distribution of Numeric Features

# In[4]:


# Plotting histogram

df.hist(figsize=(16, 16), xrot=90)

plt.show()


# #### The histogram grid above displays the distribution of the dataframe with numerical data individually. 

# ### Shape of DataFrame

# In[5]:


df.shape
print(f"It consists of", df.shape[1], "columns and", df.shape[0], "rows.")


# ### Dataset info

# In[6]:


df.info()


# #### The dataset consists of 268850 entries, six columns containing boolean datatype, eighteen columns consisting of datatype float64, and nineteen columns consisting of datatype object. 

# ### Checking Columns

# In[7]:


df.columns


# ### Describing the dataset 

# In[8]:


# Summarize the numerical features

df.describe()


# #### This shows the count of entries, mean, standard deviation, mininum, quantiles and maximun per column. 

# In[9]:


# Summarize the categorical features

df.describe(include=["object"])


# #### Column "regio1" contains the same data as column "geo_bln" indicating federal state and column "regio2" contains the same data as "geo_krs" indicating city.

# ## Data Cleaning 
# 
# #### Data cleaning involves the process of identifying inaccurate, incomplete or irrelevant parts of the data. This is carried out to prepare the data for analysis or storing in a database. It includes deleting missing values to removing irrelevant rows or columns.

# ### Renaming the columns 

# In[10]:


df.rename(columns={"regio1": "state", "regio2": "city",
                   "regio3": "neighbourhood", "date": "dateScraped"},
          inplace=True)
df.head(3)


# #### For readability, the columns; 'regio1', 'regio2', 'regio3', and 'date' are renamed to 'state', 'city', 'neighbourhood', and 'dateScraped'.

# #### Validating change

# In[11]:


df.columns


# ### Check for missing values

# In[12]:


df.isnull().sum().sort_values(ascending=False)


# #### Of the 49 columns present in the dataset, 27 columns appear to contain missing values. Further exploration of the dataset will clarify if the data is missing or the data are of boolean and string datatype. However for this project, only columns containing necessary data such as rent, service charge, living space, e.t.c will be used.

# ### Filtering (removing) columns 
# 
# #### To answer the business problem, only some of the 49 columns are needed. These columns were selected because they contain the necessary data crucial to answering the business problem.

# In[13]:


needed_columns = ["state", "city", "neighbourhood", "noRooms",
                  "livingSpace", "baseRent", "totalRent",
                  "serviceCharge", "balcony", "lift", "hasKitchen",
                  "cellar", "garden"]
df1 = df[needed_columns]

df1.head(5)


# In[14]:


df1.shape


# In[15]:


df1.dtypes


# #### A new dataFrame was created from the original dataset consisting of 268850 rows and 16 columns. The datatypes are consistent with the kind of data each column should contain. 

# ## Filtering the rows
# #### The dataset is filtered to contain only data on Brandenburg because the target area for this project is Brandenburg State.

# In[16]:


# Selecting data on only Brandenburg

df_bburg = df1[df1["state"] == "Brandenburg"].reset_index(drop=True)
df_bburg.head(5)


# In[17]:


df_bburg.shape
print(f"The new dataframe has", df_bburg.shape[0],
      "rows and", df_bburg.shape[1], "columns.")


# ### Checking for duplicates

# In[18]:


# Checking for duplicates

df_bburg.duplicated().sum()


# #### There are 355 duplicates in the subset. These duplicates will be removed.

# ### Removing the Duplicates
# #### This creates a new dataframe by default, however, adding parameter "inplace=True" makes the changes in the current dataframe and does not create another. Parameter "keep='first'" drops all duplicates except for the first occurence of each duplicate.

# In[19]:


df_bburg.drop_duplicates(keep="first", inplace=True)
df_bburg.shape

print(f"The dataframe now consists of", df_bburg.shape[1],
      "columns and", df_bburg.shape[0], "rows.")


# ### Check for missing values

# In[20]:


# Checking for missing values

df_bburg.isnull().sum()


# #### There are some missing values in the current dataframe. The "totalRent" column and the "serviceCharge" column contain missing values. Before anaylsis can proceed, the missing values have to be dropped or replaced. However, it is often better to replace the missing values with zero or NaN to indicate "no data" to python or with the mean(average), median(the middle number(s) of the data in the column) or std of the column.

# ### Handling Missing Values
# #### Numpy library is used to replace missing values in each column. 
# #### The data in the "totalRent" and "serviceCharge" column are numerical and so the missing values must be replaced with numerical values. It can be filled with zero to indicate no data. However, the mean(the average) or median of the column will be used.

# ### Checking the Median

# In[21]:


# The average(mean) of the total rent in Brandenburg;
Average1 = df_bburg["totalRent"].median()

# The average(mean) of the total rent in Brandenburg;
Average2 = df_bburg["serviceCharge"].median()

print("The median total rent in Brandenburg is ", Average1, "euros.")

print("The median service charge in Brandenburg is ", Average2, "euros.")


# In[22]:


# The maximum total rent in brandenburg

Max1 = df_bburg["totalRent"].max()

# The maximum service charge in brandenburg
Max2 = df_bburg["serviceCharge"].max()

print("The maximum total rent in Brandenburg is ", Max1, "euros.")

print("The maximum service charge in Brandenburg is ", Max2, "euros.")


# #### In this case the median total rent is significantly lower than the maximum total rent, same as the service charge. Replacing all the missing data in the total rent column with the median will create entries where the base rent is lower than the total rent which is unrealistic. It could lead to inconsistencies in the data. The best means to handle the missing data in the dataset is to remove the rows containing missing data.

# ### Removing the missing values 

# In[23]:


df_bburg.dropna(inplace=True)
df_bburg.shape


# ### Validation

# In[24]:


# Checking for missing values again;

df_bburg.isnull().sum()


# #### There are no missing values in the dataframe. 

# ###  Outliers
# #### Outliers refer to the datapoints that differ largely to other majority of the datapoints in the dataframe. The decision to remove or replace the outliers is a crucial one. Care must be taken understand the kind of data contained in the dataset and the nature of the outliers. 
# #### In real estate, it is important to consider outliers because the rental or sale value of an apartment is dependent on the apartment type, size of apartment, rent per m2 in the region, locality, closeness to the Central Business District, closeness to shops, closeness to public transport, facilities provided in the apartment, furnishing or lack thereof, amongst others. Therefore, it is normal for outliers to exist in real estate data collected within a city, especially in areas around or much further away from the city centre.

# ### Identifying the Outliers
# 
# #### The living space can be considered the independent variable, this is because the base rent, and total rent are dependent on the size of the apartment.

# In[25]:


df_bburg[["baseRent", "totalRent", "livingSpace"]].describe()


# #### This shows the count, mean, standard deviation, Q1, Q2, Q3 , and max of the "baseRent", "serviceCharge", "totalRent", and "livingSpace" columns. A breakdown of the "livingSpace" column reveals the maximum as 8684 square metres. In light of the type of data in the dataframe, it is unsual for a rental residential property to be of this size. A residential property of this size would be listed as a luxury property instead. This indicates the presence of outliers.

# #### Outliers in total rent

# In[26]:


# Plotting a boxplot to visualise the outliers in totalRent column

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_bburg["totalRent"], y=df_bburg["city"],  palette="Set2",)
sns.stripplot(x=df_bburg["totalRent"], y=df_bburg["city"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# #### Outliers in living space

# In[27]:


# Plotting a boxplot to visualise the outliers in the living space column

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (25, 25)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_bburg["livingSpace"], y=df_bburg["city"],  palette="Set2",)
sns.stripplot(x=df_bburg["livingSpace"], y=df_bburg["city"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# #### Both graphs indicate that most of the datapoints are situated at the left side of the graph, but there are a few datapoints that extend to the right side of the graph. These datapoints are the outliers. Analyzing without removing or replacing the outliers could lead to poorly formed conclusions.

# ### Using NumPy to calculate IQR 

# In[28]:


# Calculating q1 and q3 for livingSpace
Q1 = df_bburg.livingSpace.quantile(0.25)
Q3 = df_bburg.livingSpace.quantile(0.75)
Q1, Q3


# In[29]:


# Calculating q1 and q3 for totalRent
q1 = df_bburg.totalRent.quantile(0.25)
q3 = df_bburg.totalRent.quantile(0.75)
q1, q3


# #### This equals the same values for Q3 and Q1.

# In[30]:


# Calculating InterQuantileRange (iqr) for living space
IQR = Q3 - Q1

print("The Interquartile Range for the livingSpace is", IQR)

# Calculating InterQuantileRange (iqr) for total rent
iqr = q3 - q1

print("The Interquartile Range for the totalRent is", iqr)


# ### Lower Limit and Upper Limit

# In[31]:


# For livingSpace
lower_limit1 = Q1 - 1.5*IQR
upper_limit1 = Q3 + 1.5*IQR
lower_limit1, upper_limit1


# In[32]:


# For totalRent
lower_limit = q1 - 1.5*iqr
upper_limit = q3 + 1.5*iqr
lower_limit, upper_limit


# ### Identifying the Outliers

# In[33]:


# For living space
livingSpace_outliers = df_bburg[(df_bburg.livingSpace < lower_limit1)
                                | (df_bburg.livingSpace > upper_limit1)]

livingSpace_outliers.shape


# In[34]:


# For total rent
total_rent_outliers = df_bburg[(df_bburg.totalRent < lower_limit)
                               | (df_bburg.totalRent > upper_limit)]

total_rent_outliers.shape


# ### Removing the Outliers

# #### Although it is important to consider outliers in real estate data, for this project the outliers will be removed because this project is a general analysis of the region and the outliers would raise the average base rental value in the region which may lead to inaccurate conclusions. In the event that the Business Development Team determine the type of apartments to be developed, a more detailed analysis of the specific apartment type within the region will be conducted.
# #### To handle this, the entries that fall outside the upper and lower limits of the "livingSpace" column will be removed. 
# #### Check the shape of the dataframe before removing the outliers

# In[35]:


current_Shape = df_bburg.shape
current_Shape


# ### Based on the living space

# #### When removing, ensure the remaining dataframe falls within the upper and lower limit. 

# In[36]:


df_bburg_outlier_free = df_bburg[(df_bburg.livingSpace > lower_limit1)
                                 & (df_bburg.livingSpace < upper_limit1)]
df_bburg_outlier_free.shape


# In[37]:


# Plotting scatter plot for living space

plt.scatter(x=df_bburg_outlier_free["livingSpace"],
            y=df_bburg_outlier_free["livingSpace"], color="green",)
plt.show()


# ### Based on the total rent

# In[38]:


df_bburg_outlier_free1 = df_bburg[(df_bburg.totalRent > lower_limit)
                                  & (df_bburg.totalRent < upper_limit)]
df_bburg_outlier_free1.shape


# In[39]:


# Plotting scatter plot for total rent

plt.scatter(x=df_bburg_outlier_free1["totalRent"],
            y=df_bburg_outlier_free1["city"], color="blue",)
plt.show()


# #### The scatter plots above shows that compared to the total rent, removing the outliers based on the living space column produces a cleaner dataframe. When the outliers are removed based on the total rent, some outliers still remain.

# ## Data Analysis
# 
# #### Data Analysis involves the process of exploring, cleaning, transforming and modelling data with the aim of uncovering useful information, drawing informed conclusions and aiding decision-making.

# ## Business Problem 1 - The market trend in Brandenburg

# #### Brandenburg is a federal state located in the northwestern region of Germany. Its capital is Potsdam and is made up of 18 cities including Potsdam. There is need to gain an understanding of the market trend in Brandenburg and highlight the cities with the most expensive base rent and cities with the least expensive base rent. 

# ### Average Total Rent in Brandenburg

# In[40]:


# calculating the average total rent using sum and len function

def Average(x):
    avg = sum(x) / len(x)
    return avg
    total_average(x)


x = df_bburg_outlier_free["totalRent"]
average = Average(x)

print("The average total rent in Brandenburg region is " + str(round(average, 2)) + " euros")


# ### Average Base Rent Per City
# #### The base rent of an apartment is more important to the development company as this will determine the earnings and return on investment, as it is the only amount paid to the development company. Extra charges such as utilities and service charge are paid to the Utility company and facility management company respectively. 

# In[41]:


# calculating the average total rent using group by

df_bburg_average_base_rent = df_bburg_outlier_free.groupby("city").                        baseRent.agg("mean")

df_bburg_average_base_rent = pd.DataFrame(df_bburg_average_base_rent)
df_bburg_average_base_rent.sort_values(by=["baseRent"], inplace=True)

df_bburg_average_base_rent


# ### Visualization

# In[42]:


# Using Groupby to get mean base rent for each city

average_base_rent = df_bburg_outlier_free.groupby("city")["baseRent"].mean()
average_base_rent = average_base_rent.rename("mean")

# Combine using concat
city_comparsion = pd.concat([average_base_rent], axis=1)

# Plot line charts using Plotly library

fig = px.line(city_comparsion, title="Mean Base Rent per city in Brandenburg",
              color_discrete_sequence=["red"])

fig.update_layout(yaxis_title="Base Rent", legend_title="City", font_size=12)

fig.update_yaxes(rangemode="tozero")

fig.show()


# #### The chart above shows that the cities with the highest base rent are;
# #### - Dahme Spreewald Kreis 
# #### - Oberhavel Kreis 
# #### - Potsdam 
# #### The most expensive city in Brandenburg region is Potsdam. 
# 
# #### The cities with the lowest average base rent include; 
# #### - Elbe Elster Kreis 
# #### - Oberspreewald Lausitz Kreis 
# #### - Prignitz Kreis 
# #### - Uckermark Kreis 
# #### Prignitz Kreis city has the least expensive base rent. 

# ### Average Total Rent Per City
# 
# #### This is of importance to the tenants and could deter prospective tenants where the total rent is significantly higher than the base rent.

# In[43]:


# calculating the average total rent using group by

df_bburg_average_rent = df_bburg_outlier_free.groupby("city").                        totalRent.agg("mean")

df_bburg_average_rent = pd.DataFrame(df_bburg_average_rent)
df_bburg_average_rent.sort_values(by=["totalRent"], inplace=True)

df_bburg_average_rent


# ### Visualization

# In[44]:


# Plotting a bar chart

average_totalrent = df_bburg_outlier_free.groupby(["city"])["totalRent"]                    .mean().sort_index()

fig = px.histogram(x=df_bburg_outlier_free["city"].
                   value_counts().sort_index().index,
                   y=average_totalrent,
                   color=df_bburg_outlier_free["city"].
                   value_counts().sort_index().index)

fig.update_xaxes(title="City")
fig.update_yaxes(title="Average total rent per month in Brandenburg")
fig.show()


# #### Both charts produce similar results. The position of Potsdam as the capital city of Brandenburg and its proximity to Berlin State could proffer probable reasons for the high rents in the city. It is also home to a number of public and private universities that contribute to its growing student population. In comparison to the rest of Brandenburg, Potsdam, Oberhavel Kreis, and Dahme Spreewald Kreis can be considered outliers. By excluding data on these cities, the average base rent in brandenburg would be lower. 

# ### Correlation
# 
# #### Correlation (corr()) represents the relationship between columns in a dataframe. The results are measured in a range; -1 to 1.
#  #### When the result is 1, it indicates a perfect correlation. This means that the values in these columns increase and decrease simultaneously and are interdependent.
#  #### A result of 0.9 indicates a good relationship also. This works as when the result is 1.
#  #### A result of -0.9 also indicate a good relationship. However, in this case, when the values in one column rises, the value in the other column reduces.
#  #### A result of 0.01 indicates a bad relationship. This means that the values in one column cannot be predicted based on the rise or fall of values in the other column.
#  
# #### This would reveal features that are independent and interdependent on other features.

# In[45]:


df_bburg_outlier_free.corr()


# In[46]:


# Visualizing the correlations

# Set the size
plt.figure(figsize=(15, 15))

# Plotting a heatmap
sns.heatmap(df_bburg_outlier_free.corr()*100, annot=True, fmt='.0f')


# #### The living space is independent of other features. The base rent, total rent, service charge, and number of rooms are however are dependent on the living space. This means that the size of the living space determine the number of rooms, base rent, total rent, and service charge. Also, the base rent and total rent have good correlation, meaning that a rise in base rent will cause a rise in total rent. Features like balcony, lift, cellar, garden, and kitchen seems to have low correlation with the base rent, total rent, service charge and living space.

# ## Business problem 2 - The market trend in the top three most expensive cities. 
# #### This insight will aid in rental pricing and city selection of the proposed development. 

# ### Conditional Selection
# #### To answer this business problem, selecting a subset is necessary. This subset includes only rental data in Potsdam, Oberhavel Kreis, and Dahme Spreewald Kreis. It is done by dropping all rows where "city" does not equal Potsdam, Oberhavel Kreis, and Dahme Spreewald Kreis.

# ### 2.1 Oberhavel Kreis

# In[47]:


# Reducing the dataframe to a subset

Oberhavel_Kreis = df_bburg_outlier_free[df_bburg_outlier_free["city"]
                                        == "Oberhavel_Kreis"].\
                                        reset_index(drop=True)
Oberhavel_Kreis.head(3)


# In[48]:


Oberhavel_Kreis.shape


# In[49]:


Oberhavel_Kreis.describe()


# #### This shows the mean, min rent, max rent and average number of rooms. The average total rent in this city is 816.59 euros.

# In[50]:


# Plotting a bar chart

Oberhavel_Kreis_average =     Oberhavel_Kreis.groupby(["neighbourhood"])["totalRent"]    .mean().sort_index()

fig = px.histogram(x=Oberhavel_Kreis["neighbourhood"].
                   value_counts().sort_index().index,
                   y=Oberhavel_Kreis_average,
                   color=Oberhavel_Kreis["neighbourhood"].
                   value_counts().sort_index().index)

fig.update_xaxes(title="Neighbourhood")
fig.update_yaxes(title="Average rental prices in Oberhavel Kreis")
fig.show()


# #### The most expensive neighbourhood is Hohen_Neuendorf, perharps it is close to the city centre. While the least expensive neighbourhood is Schonermark, perharps it is on the outskirts of the city.

# ### 2.2 Dahme Spreewald Kreis

# In[51]:


# Reducing the dataframe to a subset

Dahme_Spreewald_Kreis =    df_bburg_outlier_free[df_bburg_outlier_free["city"]
                          == "Dahme_Spreewald_Kreis"].reset_index(drop=True)
Dahme_Spreewald_Kreis.head(3)


# In[52]:


Dahme_Spreewald_Kreis.shape


# In[53]:


Dahme_Spreewald_Kreis.describe()


# #### This shows the mean, min rent, max rent and average number of rooms. The average total rent in this city is 885.87 euros.

# In[54]:


# Plotting a bar chart

Dahme_Spreewald_Kreis_average =    Dahme_Spreewald_Kreis.groupby(["neighbourhood"])["totalRent"]    .mean().sort_index()

fig = px.histogram(x=Dahme_Spreewald_Kreis["neighbourhood"].
                   value_counts().sort_index().index,
                   y=Dahme_Spreewald_Kreis_average,
                   color=Dahme_Spreewald_Kreis["neighbourhood"].
                   value_counts().sort_index().index)

fig.update_xaxes(title="Neighbourhood")
fig.update_yaxes(title="Average rental prices in Dahme Spreewald Kreis")
fig.show()


# #### The most expensive neighbourhood is Gros_Koris, perharps it is located around the city centre. While the least expensive neighbourhood is Neu_Zauche, perharps it is located on the outskirts of the city.

# ### 2.3 Potsdam

# In[55]:


# Reducing the dataframe to a subset

Potsdam = df_bburg_outlier_free[df_bburg_outlier_free["city"]
                                == "Potsdam"].reset_index(drop=True)
Potsdam.head(3)


# In[56]:


Potsdam.shape


# In[57]:


Potsdam.describe()


# #### This shows the mean, min rent, max rent and average number of rooms. The average total rent is 985.66 euros.

# In[58]:


# Plotting a histogram

Potsdam_average = Potsdam.groupby(["neighbourhood"])["totalRent"]                    .mean().sort_index()

fig = px.histogram(x=Potsdam["neighbourhood"].
                   value_counts().sort_index().index,
                   y=Potsdam_average,
                   color=Potsdam["neighbourhood"].
                   value_counts().sort_index().index)

fig.update_xaxes(title="Neighbourhood")
fig.update_yaxes(title="Average rental prices in Potsdam")
fig.show()


# #### The most expensive neighbourhood is Forst_Potsdam_Sud. While the least expensive neighbourhood is Schlaatz.

# ### Business Problem 3 - The price per square metre in each city
# 
# #### The price per square metre is used to determine the base rent for each apartment. The living space is multiplied by the price per square metre.

# #### For Oberhavel_Kreis:

# In[59]:


# Calculating the rent per m2 for each entry

Oberhavel_Kreis["rentPerm2"] = Oberhavel_Kreis.apply(lambda x: x["baseRent"]
                                                     if x["baseRent"] < 1
                                                     else
                                                     x["baseRent"]/x
                                                     ["livingSpace"],
                                                     axis=1)


# In[60]:


'''Calculating the average rent per m2'''

Oberhavel_Kreis_mean_rentPersqm = Oberhavel_Kreis["rentPerm2"].mean()

print("The average rent per sqm in Oberhavel_Kreis is " + 
     str(round(Oberhavel_Kreis_mean_rentPersqm, 2)) + " euros")


# #### For Dahme_Spreewald_Kreis:

# In[61]:


# Calculating the rent per m2 for each entry

Dahme_Spreewald_Kreis["rentPerm2"] =     Dahme_Spreewald_Kreis.apply(lambda x: x["baseRent"]
                                if x["baseRent"] < 1
                                else x["baseRent"]/x
                                ["livingSpace"],
                                axis=1)


# In[62]:


'''Calculating the average rent per m2'''

Dahme_Spreewald_Kreis_mean_rentPersqm =     Dahme_Spreewald_Kreis["rentPerm2"].mean()

print("The average rent per sqm in Dahme_Spreewald_Kreis is " +
     str(round(Dahme_Spreewald_Kreis_mean_rentPersqm, 2)) + " euros")


# #### For Potsdam:

# In[63]:


# Calculating the rent per m2 for each entry
Potsdam["rentPerm2"] = Potsdam.apply(lambda x: x["baseRent"]
                                     if x["baseRent"] < 1
                                     else x["baseRent"]/x["livingSpace"],
                                     axis=1)


# In[64]:


'''Calculating the average rent per m2'''

Potsdam_mean_rentPersqm = Potsdam["rentPerm2"].mean()

print("The average rent per sqm in Potsdam is ",
      str(round(Potsdam_mean_rentPersqm, 2)) + " euros")


# #### The base rent for the proposed development can be calculated by multipying the living space by the rent per m2. A comparison of the cost of development and the return on investment over the next 10 years will reveal the viability of the project. Potsdam has a average rent per sqm of 11.53 euros which is higher than Dahme_Spreewald_Kreis with a average rent per sqm of 9.8 euros and Oberhavel_Kreis with a average rent per sqm of 9.36 euros. It might be more profitable to locate the proposed development in Potsdam.

# ### Business Problem 4 - Apartment size in each city
# 
# #### To determine the apartment size for the proposed development, it is helpful to gain insights on the most common apartment size.

# #### For Oberhavel_Kreis:

# In[65]:


# Using a histogram

Oberhavel_Kreis["livingSpace"].plot(kind='hist')


# #### Most apartments are between 50 sqm to 90 sqm. There are fewer apartments with living space below 40 sqm. This could indicate either low demand for apartments smaller tham 40 sqm or an exisiting gap in the market. It also means that there are more multiple room apartment, indicating a significant population of family units in the city.

# #### For Dahme_Spreewald_Kreis:

# In[66]:


# Using a histogram

Dahme_Spreewald_Kreis["livingSpace"].plot(kind='hist')


# #### Most apartments are between 50 sqm to 90 sqm. There are more apartments with living space 30 sqm than there are with 40 sqm. This could indicate the presence of student apartments or single-person studio apartments.

# #### For Potsdam:

# In[67]:


# Using a histogram

Potsdam["livingSpace"].plot(kind='hist')


# #### Most apartments are between 50 sqm to 90 sqm, with most between 60 sqm to 70 sqm. This means that the population of Potsdam consists larger of family units. There are more apartments with living space between 20 sqm to 40 sqm compared to the other cities. This could indicate a significant student population and / or single-person family units.

# ### Business Problem 5 - Common Apartment Facilities for each city
# 
# #### Having a kitchen or lift in an apartment building might seem commonplace and expected in some countries around the world, but in Germany, it is quite the opposite. It is common to rent apartments with no kitchen or lift in the building to get to apartments on higher floors. In some cases, the presence of one or more of these facilities could raise the total rent of the apartment as provisions have to be made for the maintenance of facilities such as lift. The presence of a furnished kitchen might also increase the desirability of the apartment on the market.

# #### For Oberhavel Kreis:

# In[68]:


# Loop through feature names and print each one
facilities = Oberhavel_Kreis.dtypes[Oberhavel_Kreis.dtypes == "bool"].index
for feature in facilities:
    print(feature)


# In[69]:


# Plot bar plot for each feature in Oberhavel_Kreis
for feature in facilities:
    sns.countplot(y=feature, data=Oberhavel_Kreis)
    plt.show()


# #### In this city, apartments are more likely to have a balcony or a cellar or both. The apartments are also less likely to have a lift, kitchen or a garden.

# #### For Dahme Spreewald Kreis:

# In[70]:


# Loop through feature names and print each one
facilities1 =    Dahme_Spreewald_Kreis.dtypes[Dahme_Spreewald_Kreis.dtypes == "bool"].index
for feature in facilities1:
    print(feature)


# In[71]:


# Plot bar plot for each feature in Dahme_Spreewald_Kreis
for feature in facilities1:
    sns.countplot(y=feature, data=Dahme_Spreewald_Kreis)
    plt.show()


# #### In this city, apartments are more likely to have a balcony or a cellar or both. The apartments are also less likely to have a lift or a garden, however, compared to Oberhavel Kreis, there are more apartments with a lift in this city. Apartments with or without a kitchen appear to be even distributed.

# #### For Potsdam:

# In[72]:


# Loop through categorical feature names and print each one
facilities2 = Potsdam.dtypes[Potsdam.dtypes == "bool"].index
for feature in facilities2:
    print(feature)


# In[73]:


# Plot bar plot for each categorical feature in Potsdam
for feature in facilities2:
    sns.countplot(y=feature, data=Potsdam)
    plt.show()


# #### In this city, apartments are more likely to have a balcony, a presence of a cellar and a lift appear to be almost evenly distributed. There are more apartments fitted with a kitchen than there are without a kitchen. The apartments are also less likely to have a garden. 

# ## Conclusions and Recommendations

# #### This exploratory data analysis project is lacking in many ways, one of which is a more thorough data cleaning process especially better handling of missing data and outliers. There are also more insights that could be gained from the data. Regardless, the exploratory data analysis selected a target area and was able to draw basic insights on the region.
# 
# #### From the analysis, Potsdam is the most expensive city to live in within the Brandenburg region. This is because it is the capital city of Brandenburg State and it is less than an hour train ride to the infamous Berlin State. Persons working in Berlin might choose to live in Potsdam to avoid the bustle of a major city and in the event that they are unable to secure an apartment in berlin, thus increasing demand and in turn the rental value of apartments in Potsdam. This supports the conclusion that Potsdam is home to more family units than single-person households. 
# 
# #### Dahme Spreewald Kreis and Potsdam appear to have more smaller sized apartments than Oberhavel Kreis. This could be because of a sizable student population and the possibility of young single professional working in Berlin who might chose to live in Potsdam.
# #### An exploratory data analysis on the population of Potsdam, Dahme Spreewald Kreis, and Oberhavel Kreis will provide insights on the population distribution, which will in turn lead to data-informed decisions on the viable apartments type for the proposed development. This could indicate a gap in the market for single-person households and student studio apartments.
# 
# #### Furthermore, because Potsdam has a average rent per sqm higher than Dahme_Spreewald_Kreis and Oberhavel_Kreis, it might be more profitable to locate the proposed development in Potsdam. However, the cost of construction in each city should be taken into consideration to ensure a satisfactory Return On Investment (ROI).
# #### In addition, since the total rent is determined by the base rent plus service charge and other charges, care should be taken to ensure that the service charge sufficiently covers the maintenance cost and is within the market range. Market research should be carried out on the cost of providing cleaning in general areas of the building, waste management, maintenace of lifts and the building in general. A facility management company that provides these services in a cost-effective way should be contracted for the maintenance of the proposed development. Furthermore, providing a lift, balcony, cellar and kitchen in an apartment should be considered in the proposed development as they are commonplace in the market and would increase the desirability of the proposed development. 

# ## 	References 

# #### En.wikipedia.org. 2022. Data cleansing - Wikipedia. [online] Available at: <https://en.wikipedia.org/wiki/Data_cleansing> [Accessed 16 March 2022].
# #### Bar, C., 2020. Apartment rental offers in Germany. [online] Kaggle.com. Available at: <https://www.kaggle.com/corrieaar/apartment-rental-offers-in-germany> [Accessed 1 March 2022].
# #### Sharma, P., 2021. Mastering Exploratory Data Analysis(EDA) For Data Science Enthusiasts. [online] Analytics Vidhya. Available at: <https://www.analyticsvidhya.com/blog/2021/04/mastering-exploratory-data-analysiseda-for-data-science-enthusiasts/> [Accessed 5 March 2022]
