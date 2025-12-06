# Bullying and Hate Speech Detection in Tweets
**Deep Learning Final Project**

 Hanad Ali, Abby Easterly, Saif Sarsour

 ### Introduction
 With social media platforms that require no identity verification, individuals can create fake or anonymous accounts and spread negative ideas or hate speech to other users with little to no consequences. To help negate bullying and harassment online, this program will use text data and neural networks to process statements made online and hide them from other users or even prevent the post from remaining on the platform. 

## Running instructions

We created this project in kaggle. To view it in its best form, go to:

https://www.kaggle.com/code/abby614/final-project

If kaggle is not available, or using another application, use the dictionary, dataset, and the .ipbyn above 

## Summary of Process

 ### Part 1: Data Collection and Processing
To begin, we begin data collection and processing. First, the data was found via Kaggle, where a database with tweets had already been collected and labeled as 0 - neutral, 1-negative, or 2-hateful. From this, we focused on developing a dictionary to label words numerically based on their sentiment (positive or negative). We found svveral datasets, including lists of harmful words, emojis, and general frequently used words, found on kaggle, githu, and the huggingface library. Since we did not simply want to mark a tweet as hateful wimply because of a word being present, we added the commonly used words and marked them as neutral so we could train on those words as well. We then applied our own scoring system to words, ranking words marked as positive on a scale from 1-5, 1 being slightly positive (ex: 'pretty', 'support', 'warm') and 5 being most positive (ex: 'outstanding', 'superb', 'breathtaking'), then neutral words as 0 (ex: "the", "car", "hello"). We also ranked negative word similar to the postive words, with -1 being somewhat negative (ex: 'suspect', 'leak', 'unsettled') and -5 being very negative (ex: swears, etc.). Instead of using -1:-5 for negative words, we also added -6 for slurs or violent words, intending to act as a "blacklist", or words that should immediately indicate hate speech and be flagged by the model. We created the range as we did not want tweets with negative words to be immediately flagged, like the phrase "I hate plastic plants" that is not hateful but instead an opinion, so this gives more room for the model to learn the difference between hate and negativity. 

With the dictionary created and the database of tweets uploaded, we then began cleaning the tweet text, removing things like usernames, punctuation, and urls from the strings and making all letters lowercase. We then split the string into words and created an aray of corresponding numeric values for each word (for example: "I have a kind and funny friend" becomes [0, 0, 0, 2, 0, 4, 1]). 

### Part 2: CNN Modelling
We created a Convolutional Neural Network (CNN) that combines word embeddings with our sentiment dictionary scores. First, we built a vocabulary from all tweets, mapping each unique word to a numeric ID. Each tweet is then converted into two parallel representations: word IDs (for the embedding layer) and sentiment scores (from our dictionary). The model architecture converts word IDs into 64-dimensional vectors that capture word meaning and context. It then uses our dictionary scores as an additional input channel, so the model doesn't have to learn from scratch which words are positive or negative. Three convolutional filters (sizes 2, 3, and 4) slide across the tweet to detect specific patterns (e.g., "you are stupid" or "go back to").

Linear layers flatten and classify the data, with ReLU layers for nonlinear feature extraction. Dropout (50%) prevents overfitting in the model. Finally, the output is calculated with softmax, which combines all features for final classification, with 3 possible outputs: 0 (Hate Speech), 1 (Offensive), or 2 (Neither).

The model was trained for 15 epochs using the Adam optimizer with a learning rate of 0.001 and cross-entropy loss. We used a 70/15/15 split for training, validation, and testing.

## Results and Conclusion
