#Imports
import aiml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv

# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

#Get questions & answers
questionAnswerPairs = list(csv.reader(open('chatbot-qa-pairs.csv', 'r')))
questions = [row[0] for row in questionAnswerPairs]
answers = [row[1] for row in questionAnswerPairs]

#Other global vars
stopWords = ["the","is","an","a","at","and"]

def GetMostSimilar(string, anArray):
    array = [string] + anArray
    tfidf = TfidfVectorizer(stop_words=stopWords).fit_transform(array)
    similarityArray = cosine_similarity(tfidf[0:1], tfidf)
    similarityArray = np.delete(similarityArray, 0)
    return anArray[similarityArray.argmax()]

print(GetMostSimilar("This is dogs and houses too, but also about trees", questions))


