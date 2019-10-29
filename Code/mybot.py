#Imports
import aiml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

# Welcome user
print("Welcome")

stopWords = ["the","is","an","a","at","and"]

texts = ["Trees and dogs are main characters in this story",
        "This is about airplanes and airlines",
        "This is about dogs and houses too, but also about trees",
        "Trees and dogs are main characters in this story",
        "This story is about batman and superman fighting each other", 
        "Nothing better than another story talking about airplanes, airlines and birds",
        "Superman defeats batman in the last round"]

def GetMostSimilar(string):
    array = [string] + texts
    tfidf = TfidfVectorizer(stop_words=stopWords).fit_transform(array)
    similarityArray = cosine_similarity(tfidf[0:1], tfidf)
    similarityArray = np.delete(similarityArray, 0)
    return texts[similarityArray.argmax()]

print(GetMostSimilar("This is dogs and houses too, but also about trees"))


