#!/usr/bin/env python
# coding: utf-8

# # OKCupid Dataset Use Case 2. Module 5. MAABD

# This notebook contains an analysis of a public dataset of almost 60000 online dating profiles. The dataset has been published in the [Journal of Statistics Education](http://ww2.amstat.org/publications/jse/v23n2/kim.pdf), Volume 23, Number 2 (2015) by Albert Y. Kim et al., and its collection and distribution was explicitly allowed by OkCupid president and co-founder [Christian Rudder](http://blog.okcupid.com/). Using these data is therefore ethically and legally acceptable; this is in contrast to another recent release of a different [OkCupid profile dataset](http://www.vox.com/2016/5/12/11666116/70000-okcupid-users-data-release), which was collected without permission and without anonymizing the data (more on the ethical issues in this [Wired article](https://www.wired.com/2016/05/okcupid-study-reveals-perils-big-data-science/)).

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='svg'")
from IPython.display import display,HTML
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau
import numpy as np
import math
import matplotlib.pyplot as plt

from prettypandas import PrettyPandas
sns.set(style="ticks")
sns.set_context(context="notebook",font_scale=1)

import string
import tqdm # a cool progress bar
import re
import json

import pymongo
from pymongo import MongoClient


# ### Dataset details

# The data is available at this link. The codebook includes many details about the available fields. The dataset was collected by web scraping the OKCupid.com website on 2012/06/30, and includes almost 60k profiles of people within a 25 mile radius of San Francisco, who were online in the previous year (after 06/30/2011), with at least one profile picture.
# 
# The CSV contains a row (observation) for each profile. Let's have a look at the first 10 profiles, excluding the columns whose name contains the string "essay", which contain a lot of text and are not practical at the moment.

# In[3]:


d=pd.read_csv("/home/master/UseCase_OKCupid/profiles.csv")
print("The dataset contains {} records".format(len(d)))


# In[4]:


########################################################### Database Connection and Load ############################
print('Mongo version', pymongo.__version__)
client = MongoClient('localhost', 27017)
db = client.test
collection = db.okcupid

#Import data into the database
collection.drop()


# In[5]:


# Transform dataframe to Json and store in MongoDB
records = json.loads(d.to_json(orient='records'))
collection.delete_many({})
collection.insert_many(records)


# In[6]:


#Check if you can access the data from the MongoDB.
cursor = collection.find().sort('sex',pymongo.ASCENDING).limit(1)
for doc in cursor:
    print(doc)


# In[7]:


pipeline = [
        {"$match": {"sex":"m"}},
]

aggResult = collection.aggregate(pipeline)
male = pd.DataFrame(list(aggResult))
male.head()


# In[8]:


pipeline = [
        {"$match": {"sex":"f"}},
]

aggResult = collection.aggregate(pipeline)
female = pd.DataFrame(list(aggResult))
female.head(2)


# #### Sex Distribution

# In[9]:


print("{} males ({:.1%}), {} females ({:.1%})".format(
    len(male),len(male)/len(d),
    len(female),len(female)/len(d)))


# In[10]:


# Ignore columns with "essay" in the name (they are long)
PrettyPandas(d                                   # Prettyprints pandas dataframes
   .head(10)                                    # Sample the first 10 rows
   [[c for c in d.columns if "essay" not in c]]) # Ignore columns with "essay" in the name (they are long)


# ### Age Distribution

# In[11]:


print("Age statistics:\n{}".format(d["age"].describe()))
print()
print("There are {} users older than 80".format((d["age"]>80).sum()))


# ### Find the age outliers
# Apparently we have one 110-year-old user, and only another one over-80. They might be outliers, let's have a look at their data.

# In[12]:


collection.find({"age":{ "$gt": 80 }}).count()


# In[13]:


##Let's assume the 110-year-old lady and the athletic 109-year-old gentleman (who's working on a masters program) are outliers: we get rid of them so the following plots look better. They didn't say much else about themselves, anyway.
##We then remove them
collection.delete_many({"age":{ "$gt": 80 }})
collection.find({"age":{ "$gt": 80 }}).count()
print("The dataset now contains {} records".format(collection.find({"age":{ "$lt": 80 }}).count()))


# In[14]:


cursor = collection.find().sort('sex',pymongo.ASCENDING).limit(10)
#for doc in cursor:
#    print(doc)

# I commented this because github show everything.


# In[15]:


PrettyPandas(d[d["age"]>80])


# In[16]:


# Isolate male's dataset
aggResult = collection.aggregate([{"$match": {"sex":"m"}}])
male = pd.DataFrame(list(aggResult))


# In[17]:


# Isolate female's dataset 
aggResult = collection.aggregate([{"$match": {"sex":"f"}}])
female = pd.DataFrame(list(aggResult))


# In[18]:


print("{} males ({:.1%}), {} females ({:.1%})".format(
    len(male),len(male)/len(d),
    len(female),len(female)/len(d)))


# In[19]:


d=pd.DataFrame(list(collection.find()))


# In[20]:


print("Age statistics:\n{}".format(d["age"].describe()))
print()
print("There are {} users older than 80".format((d["age"]>80).sum()))


# #### Draw age histograms for male and female users

# In[21]:


fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(10,3),sharey=True,sharex=True)
sns.distplot(male["age"], ax=ax1,
             bins=range(d["age"].min(),d["age"].max()),
             kde=False,
             color="g")
ax1.set_title("Age distribution for males")
sns.distplot(female["age"], ax=ax2,
             bins=range(d["age"].min(),d["age"].max()),
             kde=False,
             color="b")
ax2.set_title("Age distribution for females")
ax1.set_ylabel("Number of users in age group")
for ax in (ax1,ax2):
    sns.despine(ax=ax)
fig.tight_layout()


# Note that both distributions are right-skewed. Then, as is often (but not always!) the case, the mean is larger than the median.

# In[22]:


print("Mean and median age for males:   {:.2f}, {:.2f}".format(male["age"].mean(),male["age"].median()))
print("Mean and median age for females: {:.2f}, {:.2f}".format(female["age"].mean(),female["age"].median()))


# Females seem to be on average slightly older than males. Let's compare the age distributions in a single plot

# In[23]:


#########################################################################################################

fig,(ax1,ax2) = plt.subplots(nrows=2,figsize=(10,6),sharex=True)
# Plot the age distributions of males and females on the same axis
sns.distplot(male["age"], ax=ax1,
             bins=range(d["age"].min(),d["age"].max()),
             kde=False,
             color="g",
             label="males")
sns.distplot(female["age"], ax=ax1,
             bins=range(d["age"].min(),d["age"].max()),
             kde=False,
             color="b",
             label="females")
ax1.set_ylabel("Number of users in age group")
ax1.set_xlabel("")
ax1.legend()

# Compute the fraction of males for every age value
fraction_of_males=(male["age"].value_counts()/d["age"].value_counts())
# Ignore values computed from age groups in which we have less than 100 total users (else estimates are too unstable)
fraction_of_males[d["age"].value_counts()<100]=None
barlist=ax2.bar(x=fraction_of_males.index,
        height=fraction_of_males*100-50,
        bottom=50, width=1, color="gray")
for bar,frac in zip(barlist,fraction_of_males):
    bar.set_color("g" if frac>.5 else "b")
    bar.set_alpha(0.4)
ax2.set_xlim([18,70])
ax2.set_xlabel("age")
ax2.set_ylabel("percentage of males in age group")
ax2.axhline(y=50,color="k")

for ax in (ax1,ax2):
    sns.despine(ax=ax)
fig.tight_layout()


# Over-60 users are not many, but in this group there are significantly more females than males. This may be explained by the fact that, in this age group, there are more females than males in the general population.

# In[25]:


##########################################################################################################

# Age distributions of age in jointplots
sns.jointplot(male['age'], female['age'], kind="hex", stat_func=kendalltau, color="#4CB391")


# #### Study height distribution and compare with official data from the US Centers of Disease Control and Prevention ([CDC](https://www.cdc.gov/))
# We first plot the height distribution for males and females in the whole dataset

# In[26]:


fig,(ax,ax2) = plt.subplots(nrows=2,sharex=True,figsize=(6,6),gridspec_kw={'height_ratios':[2,1]})
# Plot histograms of height
bins=range(55,80)
sns.distplot(male["height"].dropna(), ax=ax,
             bins=bins,
             kde=False,
             color="g",
             label="males")
sns.distplot(female["height"].dropna(), ax=ax,
             bins=bins,
             kde=False,
             color="b",
             label="females")
ax.legend(loc="upper left")
ax.set_xlabel("")
ax.set_ylabel("Number of users with given height")
ax.set_title("height distribution of male and female users");

# Make aligned boxplots
sns.boxplot(data=d,y="sex",x="height",orient="h",ax=ax2,palette={"m":"g","f":"b"})
plt.setp(ax2.artists, alpha=.5)
ax2.set_xlim([min(bins),max(bins)])
ax2.set_xlabel("Self-reported height [inches]")

sns.despine(ax=ax)
fig.tight_layout()


# Males are (as suspected) taller than females, and the two distributions make sense.
# 
# How does this compare with general population data? Are OkCupid users maybe cheating and overreporting their height?
# 
# The CDC publishes growth charts, which contain height data for the general US population. The dataset reports statistics (3rd, 5th, 10th, 25th, 50th, 75th, 90th, 95th, 97th percentiles) for stature for different ages from 2 to 20 years. This (and more) data is plotted by the CDC in these beautiful charts.

# #### Generate a new collection to store CDC data #################
# The idea is to cross databases attributes to compare CDC versus OKCupid
# 

# In[27]:


col_cdc = db.cdcdb

#Import data into the database
col_cdc.drop()


# In[28]:


records = json.loads(pd.read_csv("https://www.cdc.gov/growthcharts/data/zscore/statage.csv").to_json(orient='records'))
col_cdc.delete_many({})
col_cdc.insert_many(records)


# In[29]:


#Check if you can access the data from the MongoDB.
cursor = col_cdc.find().limit(10)
for doc in cursor:
    print(doc)


# In[30]:


# Transform data attribute "Sex" to accomodate to OKCupid format
col_cdc.update_many({"Sex":1},{'$set':{"Sex":"m"}})
col_cdc.update_many({"Sex":2},{'$set':{"Sex":"f"}})

cdc = pd.DataFrame(list(col_cdc.find()))
cdc.head(5)


# In[31]:


cdc.tail(5)


# In[32]:


# Adjust the data to fit our format
cdc["Age"]=cdc["Agemos"]/12 # convert age in months to age in fractional years


# In[33]:


percentiles=[3,5,10,25,50,75,90,95,97]
percentile_columns=["P"+str(p) for p in percentiles] # names of percentile columns
cdc[percentile_columns]=cdc[percentile_columns]*0.393701 # convert percentile columns from centimeters to inches (ugh)
cdc20=cdc[cdc["Age"]==20].set_index("Sex") # Select the two rows corresponding to 20-year-olds (males and females)


# In[35]:


print("Height Percentiles for 20-year-old US population [inches]")
display(PrettyPandas(cdc20[percentile_columns],precision=4))


# #### Let's compare the stats for reported heights of our 20-year-olds to the CDC stats for 20-year-olds.
# 
# Note that OKCupid height data are integers, which also causes all percentiles to be integer values. To fix this, we jitter the data by ±0.5±0.5 inches by adding random uniformly distributed noise in the range [−0.5,+0.5][−0.5,+0.5] (which won't affect the mean, but will smooth the percentiles). This makes sense if we assume that users reported their height rounded to the nearest inch.

# In[36]:


mheights=male.loc[male["age"]==20,"height"] # heights of 20-year-old males
fheights=female.loc[female["age"]==20,"height"] # heights of 20-year-old females


# In[37]:


# To smooth the computation of percentiles, jitter height data by adding
# uniformly distributed noise in the range [-0.5,+0.5]
mheightsj=mheights+np.random.uniform(low=-0.5,high=+0.5,size=(len(mheights),))
fheightsj=fheights+np.random.uniform(low=-0.5,high=+0.5,size=(len(fheights),))


# In[38]:


# For each of the available percentiles in CDC data, compute the corresponding percentile from our 20-year-old users
stats=[]
for percentile,percentile_column in zip(percentiles,percentile_columns):
    stats.append({"sex":"m",
                  "percentile":percentile,
                  "CDC":cdc20.loc["m",percentile_column],
                  "users":mheightsj.quantile(percentile/100)})
    stats.append({"sex":"f",
                  "percentile":percentile,
                  "CDC":cdc20.loc["f",percentile_column],
                  "users":fheightsj.quantile(percentile/100)})
stats=pd.DataFrame(stats).set_index(["sex","percentile"]).sort_index()

# For each percentile, compute the gap between users and CDC
stats["gap"]=stats["users"]-stats["CDC"]

print("Height percentiles (in inches) for 20-year-old males")
display(PrettyPandas(stats.loc["m"],precision=4))


# In[39]:


print("Height percentiles (in inches) for 20-year-old females")
display(PrettyPandas(stats.loc["f"],precision=4))


# In[40]:


#PLOT the differences

fig,(ax1,ax2)=plt.subplots(ncols=2,sharex=True,figsize=(10,4))
#stats.loc["m"][["users","CDC"]].plot.bar(ax=ax1,color=[[0.5,0.5,1],"k"],alpha=1,width=0.8,rot=0)
stats.loc["m"][["users","CDC"]].plot.bar(ax=ax1,color=["b","grey"],alpha=1,width=0.8,rot=0)
stats.loc["f"][["users","CDC"]].plot.bar(ax=ax2,color=["g","lightgrey"],alpha=1,width=0.8,rot=0)
ax1.set_ylim([64,77])
ax2.set_ylim([58,71])
ax1.set_ylabel("Height [inches]")
ax2.set_ylabel("Height [inches]")
ax1.set_title("Height percentiles in 20y-old male users vs CDC data")
ax2.set_title("Height percentiles in 20y-old female users vs CDC data")
for ax in (ax1,ax2):
    sns.despine(ax=ax)
fig.tight_layout()


# In[41]:


#################################################################
d["height"].isnull().sum()


# #### Study how height changes with age

# In[42]:


# Investigate heights vs sex vs age
g=d.groupby(["sex","age"])["height"].mean()
fig,(ax1,ax2)=plt.subplots(ncols=2,sharex=True,figsize=(10,3))
ax1.plot(g["m"],color="g")
ax1.set_xlim(18,27)
ax1.set_ylim(69.5,71)
ax1.set(title="Average height vs age for males",
        ylabel="height",
        xlabel="age")
ax2.plot(g["f"],color="b")
ax2.set_xlim(18,27)
ax2.set_ylim(64,65.5)
ax2.set(title="Average height vs age for females",
        ylabel="height",
        xlabel="age");
for ax in (ax1,ax2):
    sns.despine(ax=ax)
fig.tight_layout()


# In[43]:


### We can also overlay CDC growth charts to the above plots, with minimal data wrangling.

cdc_m=cdc[cdc["Sex"]=="m"].groupby(np.floor(cdc["Age"]))[percentile_columns].mean()
cdc_f=cdc[cdc["Sex"]=="f"].groupby(np.floor(cdc["Age"]))[percentile_columns].mean()

# Result for males
display(PrettyPandas(cdc_m,precision=4))


# In[44]:


# Result for females
display(PrettyPandas(cdc_f,precision=4))


# In[45]:


############################################################# Compare CDC and OKCupid Percentiles #######################################################
# Compute average height per sex and age
g=d.groupby(["sex","age"])["height"].mean()

fig,(ax1,ax2)=plt.subplots(ncols=2,sharex=True,figsize=(10,5))
ax1.plot(g["m"],color="g",label="Mean of male OkCupid users")
ax1.plot(cdc_m["P75"],color="k",linestyle='dotted',label="75th percentile of male US Population")
ax1.plot(cdc_m["P50"],color="k",label="Median of male US Population")
ax1.plot(cdc_m["P25"],color="k",linestyle='dotted',label="25th percentile of male US Population")
ax1.fill_between(cdc_m.index,cdc_m["P25"],cdc_m["P75"],color="k",alpha=0.1,linewidth=0)



#ax1.legend(loc="lower right")
# Use direct labeling instead of a legend
x=cdc_m["P50"].index[-1]
ax1.text(x, g["m"].loc[:26].max(), " Mean of male OkCupid users", color="g",
         verticalalignment="bottom",fontsize="small")
ax1.text(x, cdc_m["P75"].iloc[-1]," 75th percentile of male US Population",
         verticalalignment="center",fontsize="small")
ax1.text(x, cdc_m["P50"].iloc[-1]," Median of male US Population",
         verticalalignment="center",fontsize="small")
ax1.text(x, cdc_m["P25"].iloc[-1]," 25th percentile of male US Population",
         verticalalignment="center",fontsize="small")

ax1.set_xlim(16,27)
ax1.set_ylim(67,72)
ax1.set(title="height vs age for males",
        ylabel="height [inches]",
        xlabel="age (rounded down for CDC data) [years]");
ax2.plot(g["f"],color="b",label="Mean of female OkCupid users")
ax2.plot(cdc_f["P75"],color="k",linestyle='dotted',label="75th percentile of female US Population")
ax2.plot(cdc_f["P50"],color="k",label="Median of female US Population")
ax2.plot(cdc_f["P25"],color="k",linestyle='dotted',label="25th percentile of female US Population")
ax2.fill_between(cdc_f.index,cdc_f["P25"],cdc_f["P75"],color="k",alpha=0.1,linewidth=0)

#ax2.legend(loc="lower right")
# Use direct labeling instead of a legend
x=cdc_f["P50"].index[-1]
ax2.text(x, g["f"].loc[:26].max(), " Mean of female OkCupid users", color="b",
         verticalalignment="bottom",fontsize="small")
ax2.text(x, cdc_f["P75"].iloc[-1]," 75th percentile of female US Population",
         verticalalignment="center",fontsize="small")
ax2.text(x, cdc_f["P50"].iloc[-1]," Median of female US Population",
         verticalalignment="center",fontsize="small")
ax2.text(x, cdc_f["P25"].iloc[-1]," 25th percentile of female US Population",
         verticalalignment="center",fontsize="small")

ax2.set_xlim(16,27)
ax2.set_ylim(62,67)
ax2.set(title="height vs age for females",
        ylabel="height [inches]",
        xlabel="age (rounded down for CDC data) [years]");
for ax in (ax1,ax2):
    sns.despine(ax=ax)
fig.tight_layout()


# #### How do users self-report their body type?
# An interesting categorical attribute body_type contains the self-reported body type of the user, chosen from a limited set of options.

# In[46]:


fig,ax=plt.subplots(figsize=(6,5))
sns.countplot(y="body_type",hue="sex",
              order=d["body_type"].value_counts().sort_values(ascending=False).index,
              data=d,palette={"m":"g","f":"b"},alpha=0.5,ax=ax);
ax.set_title("Number of female and male users self-reporting each body type")
sns.despine(ax=ax)


# In the plot above, males and females are two sub-groups of the population, whereas body_type is a categorical attribute. It is interesting to compare how users in each of the two sub-groups (i.e. males and females) are likely to use each of the available categorical values; this is normally done through contingency tables.

# In[47]:


# Define visualization function
def compare_prevalence(series,g1,g2,g1name,g2name,g1color,g2color,ax):
    
    # for each categorical value represented in series, number of users in group g1 which have this value
    g1n=series.loc[g1].value_counts()
    # for each categorical value represented in series, number of users in group g2 which have this value
    g2n=series.loc[g2].value_counts()
    
    # join the two series in a single dataframe, filling 0 where indices don't match
    # (e.g. if a value represented in g1 did never appear in g2)
    df=pd.concat({"g1n":g1n,"g2n":g2n},axis=1).fillna(0)
    # df has one row for every distinct value of series in the union of g1 and g2
    
    # normalize the data
    df["g1f"]=df["g1n"]/(df["g1n"].sum()) # fraction of g1 users with each categorical value
    df["g2f"]=df["g2n"]/(df["g2n"].sum()) # fraction of g2 users with each categorical value
    
    assert(math.isclose(df["g1f"].sum(),1)) 
    assert(math.isclose(df["g2f"].sum(),1))
    
    # for each row of df, we now compute how frequent the value was in g1 compared to the frequency it had in g2.
    df["frac12"]=df["g1f"]/(df["g1f"]+df["g2f"])
    # we expect df["frac12"] to be 0.5 for values that were equally frequent in g1 and g2 (note that this does not depend on the size of g1 and g2)
    # we expect df["frac12"] to be 0 for values that were only seen in g2 and never seen in g1
    # we expect df["frac12"] to be 1 for values that were only seen in g1 and never seen in g2
    
    df=df[(df["g1n"]+df["g2n"])>=50] # exclude values which are too rare
    df=df.sort_values("frac12")
    
    # Draw the left bars
    ax.barh(y=range(len(df)),
                width=df["frac12"],
                left=0,
                height=1,
                align="center",
                color=g1color,alpha=1)
    # Draw the right bars
    ax.barh(y=range(len(df)),
                width=df["frac12"]-1,
                left=1,
                height=1,
                align="center",
                color=g2color,alpha=1)
    
    # Draw a faint vertical line for x=0.5
    ax.axvline(x=0.5,color="k",alpha=0.1,linewidth=5)
    ax.set(xlim=[0,1],
           ylim=[-1,len(df)-0.5],
           yticks=range(len(df)),
           yticklabels=df.index,
           xlabel="fraction of users",
           ylabel=series.name)
    
    ax.set_title("Relative prevalence of {} ($n={}$) vs {} ($n={}$)\nfor each value of {}".format(
                g1name,g1.sum(),g2name,g2.sum(),series.name),
                loc="left",fontdict={"fontsize":"medium"})
    ax.text(0.02,len(df)-1,g1name,verticalalignment="center",horizontalalignment="left",size="smaller",color="w")
    ax.text(0.98,0,g2name,verticalalignment="center",horizontalalignment="right",size="smaller",color="w")

    def color_for_frac(f):
        # Blend g1color and g2color according to f (convex linear combination):
        # 0 returns g1color, 1 returns g2color)
        ret=np.array(g1color)*f+np.array(g2color)*(1-f)
        if(np.linalg.norm(ret)>1):          # If the resulting rgb color is too bright for text,
            ret=(ret/np.linalg.norm(ret))*1 # rescale its brightness to dark (but keep hue)
        return ret
        
    for i,tl in enumerate(plt.gca().get_yticklabels()):
        tl.set_color(color_for_frac(df["frac12"].iloc[i]))
        
    sns.despine(ax=ax,left=True)

# Apply visualization function 
fig,ax = plt.subplots(figsize=(10,3))
compare_prevalence(
    series=d["body_type"],                          # Which categorical attribute?
    g1=d["sex"]=="m",      g2=d["sex"]=="f",        # Definition of the two groups
    g1name="male users",   g2name="female users",   # Names of the two groups
    g1color=[0.5,0.5,1.0], g2color=[1.0,0.5,0.5],   # Colors for the two groups
    ax=ax)
fig.tight_layout()


# ### Analyzing essays
# ##### The data contains essays written by the users on the following topics:
# - essay0: My self summary
# - essay1: What I’m doing with my life
# - essay2: I’m really good at
# - essay3: The first thing people usually notice about me
# - essay4: Favorite books, movies, show, music, and food
# - essay5: The six things I could never do without
# - essay6: I spend a lot of time thinking about
# - essay7: On a typical Friday night I am
# - essay8: The most private thing I am willing to admit
# - essay9: You should message me if...

# In[48]:


# In the following, we concatenate all essays to a single string and ignore the different themes.
d["essays"]=""
for f in ["essay"+str(i) for i in range(10)]:
    d.loc[d[f].isnull(),f]=""
    d["essays"]=d["essays"]+" "+d[f]


# In[49]:


# Let's index and count all unique words in all essays.
from collections import Counter
wordcounts=Counter()
for e in d["essays"].str.split():
    wordcounts.update([w.strip(string.whitespace+string.punctuation) for w in e])


# In[50]:


for w,c in wordcounts.most_common(100):
    print(c,w)


# In[51]:


# Among tha 10k most common words, we filter those with at least 4 characters. We build a dataframe that contains a binary column 
# for each of these and a row for each user. The value will be True where if the user's essay contains the word, False otherwise.

#import tqdm # a cool progress bar

# Let's consider the most common 10k words
#words=[w for w,c in wordcounts.most_common(10000) if len(w)>=4 and w.isalpha()]

words=[w for w,c in wordcounts.most_common(100) if len(w)>=4 and w.isalpha()]


# In[53]:


d_contains=pd.DataFrame(index=d.index)
# This operation takes a lot of time
for w in tqdm.tqdm(words):
    d_contains[w]=d["essays"].str.contains("\\b"+w+"\\b")


# In[54]:


# These frequent words were part of the html and we should ignore them
d_contains=d_contains.drop("href",axis=1)
#d_contains=d_contains.drop("ilink",axis=1)
d_contains.to_pickle("d_contains.pickle") # Save a cache

print("The dataset contains {} rows (users) and {} columns (words)".format(
        len(d_contains.index),len(d_contains.columns)))


# #### Let's visualize the resulting dataframe.

# In[55]:



#We show the transpose of the dataframe, i.e. one row per word and one column per user. 
#We only display 100 users (i.e. less than one hundreth of all users in the dataset), and 100 words 
#(i.e. about 1/80th of all the frequent words we found)

fig,(ax1,ax2)=plt.subplots(nrows=2,sharex=True,figsize=(10,15))
sns.heatmap(d_contains.iloc[0:100,0:49].transpose(),
            ax=ax1,cbar=None)
sns.heatmap(d_contains.iloc[0:100,-49:-1].transpose(),
            ax=ax2,cbar=None)
ax1.set_title("Which of the first 100 users (columns) use which of the 50 most frequent words (rows)")
ax2.set_title("Which of the first 100 users (columns) use which of the 50 least frequent words (rows) among the "+str(d_contains.shape[1])+" most frequent ones")
for ax in (ax1,ax2):
    ax.set_xticks([])
    ax.set_xlabel("Users")
    ax.set_ylabel("User's essays contain word")
fig.tight_layout()


# In[57]:


print(d["essays"].str.contains("\\bbinding of isaac\\b").sum())
print(d["essays"].str.contains("\\bisaac asimov\\b").sum())
print(d["essays"].str.contains("\\basimov\\b").sum())
d["essays"].str.extract("(\\bisaac [a-z]*\\b)").dropna().value_counts()


# ####  Mongo Queries Essays Patters 

# In[58]:


#Check if you can access the data from the MongoDB. For example, we focus on essay5, but other essays can be examined
cursor = collection.find().limit(50)
#for doc in cursor:
#    print(doc["essay5"])


# In[59]:


### Stablishing patters to search and find common areas of interest between males and fmales
pattern = "likes dogs"
pattern3 = "sports" 
pattern2 = "family" 
pat = re.compile(pattern3, re.I)
   
pipeline1 = [{"$match": {"education":"graduated from college/university" , 'essay5': {'$regex': pat}}}]

pipeline2 = [{"$match": {"speaks":"english (fluently)" , 'essay5': {'$regex': pat}}}]

pipeline3 = [{"$match": {"speaks":"spanish" , 'essay5': {'$regex': pat}}}]


# In[60]:


aggResult = collection.aggregate(pipeline1)
dfe = pd.DataFrame(list(aggResult))
dfe.head()


# In[61]:


# Explore other specific patterns
dfe["essay5"].str.extract("(\\bfootball [a-z]*\\b)").dropna()

