#Imports
import aiml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

# Welcome user
print("Welcome")

stopWords = ["the","is","an","a","at","and"]

def TermFrequencyTable(term):
    frequencyDict = dict()
    
    for word in term.split():
        if word in frequencyDict:
            frequencyDict[word] += 1
        else:
            frequencyDict[word] = 1
            
    return frequencyDict

texts = ["This is about airplanes and airlines",
        "This is about airplanes and airlines",
        "This is about dogs and houses too, but also about trees",
        "Trees and dogs are main characters in this story",
        "This story is about batman and superman fighting each other", 
        "Nothing better than another story talking about airplanes, airlines and birds",
        "Superman defeats batman in the last round"]

tfidf = TfidfVectorizer(stop_words=stopWords).fit_transform(texts)
#print(tfidf)

similarity = cosine_similarity(tfidf[0:1], tfidf)

print(similarity)
