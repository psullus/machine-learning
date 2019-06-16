# Machine Learning Engineer Nanodegree
## Capstone Proposal
Patrick O'Sullivan  
June 16th, 2019

## Proposal

Business Public Sentiment

### Domain Background

Businesses are in every aspect of our lives from the moment we are born (and earlier) to the moment we die (and after). They have huge control over us and can manipulate us in may way (and often do). A number of business have build platforms that allow there customers rate there one-off experiences with a business but what about the overall sentiment of a business.

Understanding the overall sentiment of a business may help us make a more informed decision about which business we want to use for a given service and hence encourage businesses to be more conscience and pro-active about there public sentiment.

There is a number of news articles on this topic highlight its importance:  
* https://www.forbes.com/sites/jiawertz/2018/11/30/why-sentiment-analysis-could-be-your-best-kept-marketing-secret/#91f358e2bbec
* https://www.businessinsider.com/negative-social-media-sentiment-hurts-sales-2013-6?r=US&IR=T
* https://www.theguardian.com/news/datablog/2013/jul/15/reputation-management-business-swallow-bitter-pill
* https://www.business2community.com/branding/measuring-corporate-sentiment-02091306

My personal motivation for working on sentiment is to understand the importance of how what we say and do effects how people perceive us. I'm starting with businesses but "us" could be a team or a person also.

### Problem Statement

The main objective of the project will be to use Machine Learning to decide the sentiment of text.

When give a string of text we want to be able to say whether the sentiment of the text is considered positive or negative.

### Datasets and Inputs

For this project we will use  a dataset  called [Sentiment140](http://help.sentiment140.com/for-students/). The dataset is split in both a training and testing set. The training set contains 1600000 tweets.

The tweets are is a csv file with the following fields: 
* id
* date
* query
* user
* text

The tweets are classified as
* 0 = negative
* 2 = neutral
* 4 = positive

I can use the twitter [Standard search API](https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets) to get real data as input.

### Solution Statement

The proposed solution to this problem is to us the Natural Language Toolkit (NLTK) and Machine Learning technique that have proved to be successful in the classification of sentiment.

First we will read the dataset (see Dataset section above) and do any pre-processing that is needed to make sure the data is as clean as possible. Then we will split the training and test set and build and compile our model, then evaluate and validate the accuracy of our model and finally get a prediction and accuracy score.

### Benchmark Model

There are a number of projects on kaggle in this area.
* [twitter-sentiment-analysis](https://www.kaggle.com/paoloripamonti/twitter-sentiment-analysis)
* [python-nltk-sentiment-analysis]((https://www.kaggle.com/ngyptr/python-nltk-sentiment-analysis)

Depending on my final solution I will use one of these projects to compare my accuracy score too.

### Evaluation Metrics

The evaluation metric for this project is an accuracy score.

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

### References
1. Sentiment140: [http://help.sentiment140.com/for-students/](http://help.sentiment140.com/for-students/)
2. Twitter Search API: [https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets](https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets)
3. Kaggle: [https://www.kaggle.com](https://www.kaggle.com)