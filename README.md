# BCG-MovieLens-Python
Python Project for BCG Rise 2.0 

This is the python project from my BCG Rise Course

We were given 3 datasets from the MovieLens dataset. Our task is to perform Exploratory Data Analysis (EDA) and find the indepedent variables that affect the target Variable(Ratings in this case)

These were the tasks given

Problem Objective :
Here, we ask you to perform the analysis using the Exploratory Data Analysis technique. You need to find features affecting the ratings of any particular movie and build a model to predict the movie ratings.
Domain: Entertainment
Analysis Tasks to be performed:
•	Import the three datasets
•	Create a new dataset [Master_Data] with the following columns MovieID Title UserID Age Gender Occupation Rating. (Hint: (i) Merge two tables at a time. (ii) Merge the tables using two primary keys MovieID & UserId)
•	Explore the datasets using visual representations (graphs or tables), also include your comments on the following:
1.	User Age Distribution
2.	User rating of the movie “Toy Story”
3.	Top 25 movies by viewership rating
4.	Find the ratings for all the movies reviewed by for a particular user of user id = 2696
•	Feature Engineering:
            Use column genres:
1.	Find out all the unique genres (Hint: split the data in column genre making a list and then process the data to find out only the unique categories of genres)
2.	Create a separate column for each genre category with a one-hot encoding ( 1 and 0) whether or not the movie belongs to that genre. 
3.	Determine the features affecting the ratings of any particular movie.

Dataset Description :
These files contain 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens in 2000.

Ratings.dat
    Format - UserID::MovieID::Rating::Timestamp
Field	Description
UserID	Unique identification for each user
MovieID	Unique identification for each movie
Rating	User rating for each movie
Timestamp	Timestamp generated while adding user review
•	UserIDs range between 1 and 6040 
•	The MovieIDs range between 1 and 3952
•	Ratings are made on a 5-star scale (whole-star ratings only)
•	A timestamp is represented in seconds since the epoch is returned by time(2)
•	Each user has at least 20 ratings
 
Users.dat
Format -  UserID::Gender::Age::Occupation::Zip-code
Field	Description
UserID	Unique identification for each user
Genere	Category of each movie
Age	User’s age
Occupation	User’s Occupation
Zip-code	Zip Code for the user’s location
All demographic information is provided voluntarily by the users and is not checked for accuracy. Only users who have provided demographic information are included in this data set.
•	Gender is denoted by an "M" for male and "F" for female
•	Age is chosen from the following ranges:
 
Value	Description
1	"Under 18"
18	"18-24"
25	"25-34"
35	"35-44"
45	"45-49"
50	"50-55"
56	"56+"
 
•	Occupation is chosen from the following choices:

Value
 		Description
0		"other" or not specified
1		"academic/educator"
2		"artist”
3		"clerical/admin"
4		"college/grad student"
5		"customer service"
6		"doctor/health care"
7		"executive/managerial"
8		"farmer"
9		"homemaker"
10		"K-12 student"
11		"lawyer"
12		"programmer"
13		"retired"
14		 "sales/marketing"
15		"scientist"
16		 "self-employed"
17		"technician/engineer"
18		"tradesman/craftsman"
19		"unemployed"
20		"writer”

Movies.dat
Format - MovieID::Title::Genres
Field	Description
MovieID	Unique identification for each movie
Title	A title for each movie
Genres	Category of each movie
 
•	 Titles are identical to titles provided by the IMDB (including year of release)
 
•	Genres are pipe-separated and are selected from the following genres:
1.	Action
2.	Adventure
3.	Animation
4.	Children's
5.	Comedy
6.	Crime
7.	Documentary
8.	Drama
9.	Fantasy
10.	Film-Noir
11.	Horror
12.	Musical
13.	Mystery
14.	Romance
15.	Sci-Fi
16.	Thriller
17.	War
18.	Western

Exploratory Data Analysis (EDA)

## Importing the necessary libraries 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

## Importing movie dataset .dat
dfmovies = pd.read_csv(
    'C:/Users/Jonathan Jie/My Python Stuff/movies.dat',
    delimiter='::',
    header=None,
    names=['MovieID', 'Title', 'Genres'],
    engine='python',
    encoding='ISO-8859-1'
)

## Checking for proper import
dfmovies.head()

![Movies head](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/d28a0117-58c5-47c2-a2d2-3d879ce98564)

## Exploratory Data Analysis on dfmovies - Check for number of records in each column, data having null or not null, Data type
dfmovies.info()

![moviesinfo](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/2753aba7-4894-4b54-9e08-74b0421d35c8)

check for Duplication
dfmovies.nunique()

Missing Values Calculation
dfmovies.isnull().sum

![movies isnull](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/87658b96-9de1-4ef1-a5c4-07f21db08999)

Repeat import and EDA for 2 more datasets

## Each CSV seems clean so no further need for data cleaning.
## Time to move onto merging into a single Data Frame.
##First Merge on Movie ID
movie_ratings = pd.merge(dfmovies,dfratings,on = 'MovieID')

## Second merge to get Master_Data Data Frame
Master_Data = pd.merge(movie_ratings,dfusers, on = 'UserID')

## Trim the Data frame of unused columns
Master_Data = Master_Data[['MovieID', 'Title', 'UserID', 'Age', 'Gender', 'Occupation', 'Rating','Zip-code','Genres']]

![Master_Data dataframe](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/79971c39-7255-4dab-be88-cdec0b8ee468)

Exploring the dataset - visualization
##  User Rating Distribution

Plot the user rating distribution
ratings = Master_Data['Rating']

Create the histogram with custom bin sizes
bin_edges = [1, 2, 3, 4, 5]

plt.figure(figsize=(8, 6))
hist, bins, _ = plt.hist(ratings, bins=bin_edges, color='lightcoral', edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('User Rating Distribution')

Calculate the percentage of each rating
total_ratings = len(ratings)
percentage = [(count / total_ratings) * 100 for count in hist]

Calculate the midpoints of each bin
bin_midpoints = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
Annotate the bars with both count and percentage
for i, (count, percent) in enumerate(zip(hist, percentage)):
    plt.annotate(f'{int(count)}\n({percent:.1f}%)', xy=(bin_midpoints[i], count), ha='center', va='bottom')

![User Rating Distribution Histogram](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/06236013-56ce-4670-acd4-6894d35d7784)

## Gender Distribution

gender_counts = dfusers['Gender'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct=lambda p:f'{p:.1f}% ({int(p*sum(gender_counts)/100)})', startangle=140)
plt.title("Gender Distribution")
plt.axis('off')

![Gender Distribution](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/b0c5ba6d-e8c2-437f-abb6-0a5531443ba5)

## Users by occupation
![Users by Occupation](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/2c6d2125-0024-4549-bbb0-32a118806a22)

## Qn 1 (User Age Distribution)Histogram of Age Distribution in seaborn

from matplotlib.ticker import FuncFormatter

Age values from the "Master_Data" DataFrame
age_data = dfusers['Age']

Create the countplot using Seaborn
fig, ax = plt.subplots(figsize=(8, 6))  # Use different variable names for fig and ax
age_dist = sns.countplot(
    data=dfusers,
    x='Age',
    palette='coolwarm',
    ax=ax  # Specify the axis for the plot
)
plt.title('Age Distribution', fontdict={'fontsize': 14, 'weight': 'bold'})
plt.xlabel('Age Group', fontdict={'fontsize': 12})
plt.ylabel('No. of Users', fontdict={'fontsize': 12})

Define a function to format y-axis labels with thousands separators
def format_thousands(x, pos):
    return '{:,.0f}'.format(x)

Apply the y-axis label formatting function
age_dist.yaxis.set_major_formatter(FuncFormatter(format_thousands))

Define the x-axis tick positions and labels
x_positions = [0, 1, 2, 3, 4, 5, 6]
x_labels = ['Under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']

Set the x-axis ticks and labels
plt.xticks(x_positions, x_labels)


Add data labels to the bars
for container in age_dist.containers:
    age_dist.bar_label(container, padding=2, fmt='{:,.0f}')
![Users Age Distribution](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/463904c9-7da3-4aed-9aca-48279e766be2)

## Qn 2 (User rating of the movie “Toy Story”)

Extracting just the data for Toy Story 1995 only. Not toy story 2
toy_story_ratings = Master_Data[Master_Data['Title'] == 'Toy Story (1995)']

Finding the average rating for toy story
average_rating = toy_story_ratings['Rating'].mean()

plt.figure(figsize=(8, 6))
sns.countplot(data=toy_story_ratings, x='Rating')
plt.xlabel('Rating')
plt.ylabel('Count of User IDs')
plt.title('User Ratings for "Toy Story"')

ax = plt.gca()
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
    ![Toy Story Ratings](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/f3556a63-445c-4d2d-8247-4dc84783cb49)

## Qn 3 Top 25 movies by viewership rating - method 1 by count of ratings i.e. how many people viewed it

rating_counts = Master_Data.groupby('Title')['Rating'].count()
top_25_views = rating_counts.sort_values(ascending=False).head(25)
![Top 25 Movies by Rating Count](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/af20be90-ac3c-4caf-8ece-1a5a3f36c130)

plt.figure(figsize=(12, 6))  
ax = sns.barplot(
    x=top_25_views.values,
    y=top_25_views.index,
    palette=('YlOrRd')
)
plt.xlabel('Number of Ratings') 
plt.ylabel('Movie Title')       
plt.title('Top 25 Movies by Viewership Rating')

for p in ax.patches:
    ax.annotate(f'{p.get_width():.0f}', (p.get_width(), p.get_y() + p.get_height() / 2), ha='left', va='center')

plt.xticks([]) 
plt.tight_layout()            
plt.show()

![Top 25 Movies by Rating Count Graph](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/3b7cd5c7-4548-44f4-a4b4-4b5792aba650)

## Qn 3 Top 25 movies by viewership rating - method 2 Highest average rating

rating_average = Master_Data.groupby('Title')['Rating'].mean()
top_25_average = rating_average.sort_values(ascending=False).head(25)
![Top 25 Movies by Average Rating](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/ade11340-1166-41ff-9b65-3d44aca62682)

Trying to use a distribution plot for the density of ratings.

sns.distplot(top_25_average)
plt.title('Distribution of Average Rating for Top 25 Movies')
plt.xlabel('Average Rating')
plt.ylabel('Density')
![Distribution of Average Rating for Top 25 movies by average rating](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/9f778c54-fd85-4e3e-a5dc-e3869d93a6ab)

## Qn 4 Find the ratings for all the movies reviewed by for a particular user of user id = 2696.
creating a new dataframe to extract all the data related to UserID 2696
ratings_user_2696 = Master_Data[Master_Data['UserID'] == 2696]

ratings_user_2696_sorted = ratings_user_2696.sort_values(by='Rating', ascending=False)
![Movie Ratings by User 2696](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/18e930ef-a632-4a95-8c85-e830a96ef3cb)

Countplot of User 2696 Distribution of ratings given by the user.
sns.countplot(x='Rating', data=ratings_user_2696)
plt.title('Distribution of Ratings by User 2696')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
![Distribution of Ratings by User 2696](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/46ed682e-e2bc-41ff-9ce3-299de2fa6eb5)

## Heatmap sorted by highest to lowest rating

average_ratings = ratings_user_2696.groupby('Title')['Rating'].mean().sort_values(ascending=False)
sorted_movie_titles = average_ratings.index

heatmap_data = ratings_user_2696.pivot_table(index='Title', columns='MovieID', values='Rating')
heatmap_data = heatmap_data.reindex(index=sorted_movie_titles)
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.1f', linewidths=0.5)
plt.title('Movie Ratings Heatmap by User 2696 (Sorted by Descending Ratings)')
plt.xlabel('MovieID')
plt.ylabel('Movie Title')
plt.xticks(rotation=90)

![Movie Heat Map by User 2696](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/611dd915-e468-47cb-8455-a8835f2637e1)

##Feature Engineering Qn 1. Find out all the unique genres 

Split the 'Genres' column into a list of genres
Master_Data['Genre'] = Master_Data['Genres'].str.split('|')

Flatten the list to get all unique genre values
all_genres = [genre for genres_list in Master_Data['Genre'] for genre in genres_list]

Convert the unique genre values into a set to remove duplicates
unique_genres = set(all_genres)

Convert the set back to a list if needed
unique_genres_list = list(unique_genres)

Print or use unique_genres_list as needed
print(unique_genres_list)
![unique genres](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/4b845c85-8dfa-4a72-8e66-771d4590043a)

Dropped the 'Genre' column now
Master_Data.drop(['Genre'], axis=1, inplace=True)

## perform one hot encoding
genres_encoded = Master_Data['Genres'].str.get_dummies('|')

Master_Data = pd.concat([Master_Data, genres_encoded], axis=1)

Master_Data.drop(columns=['Genres'], inplace=True)

![One Hot Encoding for Genres](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/e4043378-193a-44ac-9794-0b8a5b75be18)

## Running a Chi Square test for fit
from scipy.stats import chi2_contingency

chi_square_results = pd.DataFrame(columns=['Variable', 'Chi-Square Statistic', 'P-Value'])

hi Square Test - For significant association or independence between them

chi_square_results_list = []

categorical_columns = ['MovieID','Title','UserID','Age','Gender', 'Occupation', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary',
                       'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

ratings_column = 'Rating'

for column in categorical_columns:
    contingency_table = pd.crosstab(Master_Data[column], Master_Data[ratings_column])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    # Append the results as a dictionary to the list
    chi_square_results_list.append({'Variable': column, 'Chi-Square Statistic': chi2, 'P-Value': p})

Create a DataFrame from the list
chi_square_results = pd.DataFrame(chi_square_results_list)

Print the DataFrame with the results

![Chi Square Results](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/4dc4e3ea-d115-4c43-a172-bfd0d4035f39)
![Chi Square Viz](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/82a5be64-9686-4678-bf86-73794892c0f1)

## Spearman test - Assess the strength and direction of the ordinal or continuous variables monotonic relationship

spearman_data = Master_Data[['MovieID', 'Title', 'UserID', 'Age', 'Gender', 'Occupation','Rating', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western

import scipy.stats as stats 

# Create your spearman_data DataFrame here (contains the variables you want to analyze)

correlation_coeffs = []
p_values = []
variable1_list = []
variable2_list = []

for column1 in spearman_data.columns:
    for column2 in spearman_data.columns:
        if column1 != column2:
            # Calculate Spearman correlation and p-value
            rho, p_value = stats.spearmanr(spearman_data[column1], spearman_data[column2])
            correlation_coeffs.append(rho)
            p_values.append(p_value)
            variable1_list.append(column1)
            variable2_list.append(column2)

Create a DataFrame to store the results
spearman_results = pd.DataFrame({
    'Variable 1': variable1_list,
    'Variable 2': variable2_list,
    'Spearman Coefficient': correlation_coeffs,
    'P-Value': p_values
})
![Spearman Coefficient](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/32f0cdc6-9c5d-469e-ae5e-ed76d9506194)

## Plotting the correlation Matrix 
Set the figure size
plt.figure(figsize=(12, 8))

Create a heatmap of the correlation matrix with custom annotations
for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        plt.text(j + 0.5, i + 0.5, f"{correlation_matrix.iloc[i, j]:.2f}", ha='center', va='center', fontsize=8)

Create the heatmap with a color bar
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, cbar=True, linewidths=0.5)

Customize the plot (optional)
plt.title('Spearman Correlation Matrix')
plt.xticks(rotation=90)
plt.yticks(rotation=0)

![Spearman Correlation Matrix](https://github.com/JonathanJamesJie/BCG-MovieLens-Python/assets/139092596/533afdd7-f566-429c-a9f2-75356e479c5c)

'''
Overall, these correlations suggest that user age, certain movie genres (e.g., drama and film-noir), 
and possibly user occupation have some influence on user ratings. 
However, the correlations are generally weak to moderate, 
indicating that these factors alone may not explain a significant portion of the variance in ratings. 
'''
