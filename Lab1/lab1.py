import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random

from nltk.util import pr



#nltk.download('twitter_samples')

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# print('Number of positive tweets: ',len(all_positive_tweets))
# print('Number of negative tweets: ',len(all_negative_tweets))

# print('\nThe type of all_positive_tweers is: ',type(all_positive_tweets))
# print('The type of a tweet entry is: ',type(all_negative_tweets[0]))

fig = plt.figure(figsize=(5,5))

labels = 'ML_BSB_LEC','ML_HAP_Lec','ML_HAP-Lab'

sizes = [40,35,25]

plt.pie(sizes,labels=labels,autopct='%.2f%%',shadow=True,startangle=90)

#Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')

#Displaying the chart.
#plt.show()

#Custom size for a figure
fig = plt.figure(figsize=(5,5))

labels = 'Positives','Negatives'

sizes = [len(all_positive_tweets),len(all_negative_tweets)]

plt.pie(sizes,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
plt.axis('equal')
#plt.show()

#print positive in green
#print('\033[92m'+all_positive_tweets[random.randint(0,5000)])

#print negative in red
#print('\033[91m'+all_negative_tweets[random.randint(0,5000)])

tweet = all_positive_tweets[2277]
#print(tweet)

# nltk.download('stopwords')

# --------------------------------------------------------------------------------------------#
#-----------------------Tokenizing Word--------------------------------------------------#


import re                                   #module for regex
import string                               #for string operations

from nltk.corpus import stopwords           #module for stop words that come with NLTK
from nltk.stem import PorterStemmer         #module for stemming
from nltk.tokenize import TweetTokenizer    #module for tokenizing strings

#print('\033[92m'+tweet)
#print('\033[94m')

#remove hyperlinks sub() to substitue value 
tweet2 = re.sub(r'https?:\/\/.*[\r\n]*','',tweet);

#remove hashtags
tweet2 = re.sub(r'#','',tweet2)

# print(tweet2)

print()
# print('\033[92m'+tweet)
# print('\033[94m')

tokenizer = TweetTokenizer(preserve_case=False)

tweet_tokens = tokenizer.tokenize(tweet2)

print()
print('tokenized String:')
print(tweet_tokens)

#---------------------------------Remove stop words and punctuations----------------------------------------------------#

stopwords_english = stopwords.words('english')

print('Stop Words\n')
print(stopwords_english)

print('\nPunctuation\n')
print(string.punctuation)


print()
print('\033[92m')
print(tweet_tokens)
print('\033[94m')

tweet_clean = []

for word in tweet_tokens:
    if(word not in stopwords_english and word not in string.punctuation):
        tweet_clean.append(word)
    
print('removed stop words and punctuations')
print(tweet_clean)

print()
print('\033[92m')
print(tweet_clean)
print('\033[94m')


#We can see that the prefix happi is more commonly used. We cannot choose happ because it is the stem of
# unrelated words like happen.
# NLTK has different modules for stemming and we will be using the PorterStemmer
stemmer = PorterStemmer()

tweet_stem = []

for word in tweet_clean:
    stem_word = stemmer.stem(word) #stemming a word
    tweet_stem.append(stem_word)

print('stemmed words:')
print(tweet_stem)




