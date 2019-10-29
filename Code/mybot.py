#Imports
import aiml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import sys

# Create a Kernel object & bootstrap to aiml file.
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="chatbot-aiml.xml")

#Get questions & answers
questionAnswerPairs = list(csv.reader(open('question-and-answer-pairs.csv', 'r')))
breedAndInfoPairs = list(csv.reader(open('breed-and-info-pairs.csv', 'r')))

#Global vars
stopWords = ["the","is","an","a","at","and"]
waysOfSayingYes = ["yes", "y", "correct", "affirmative", "okay", "ok", "right", "of course", "by all means", "sure", "indeed", "yea", "yeah", "yep", "yup",
                   "certainly"]
questions = [row[0] for row in questionAnswerPairs]
answers = [row[1] for row in questionAnswerPairs]
breeds = [row[0] for row in breedAndInfoPairs]
breedInfo = [row[1] for row in breedAndInfoPairs]

# GetMostSimilar : Checks input list for most similar string to input string
#   returns # if no string found
def GetMostSimilar(string, searchArray, similariyValue=0):
    array = [string] + searchArray
    tfidf = TfidfVectorizer(stop_words=stopWords).fit_transform(array)
    similarityArray = cosine_similarity(tfidf[0:1], tfidf)
    similarityArray = np.delete(similarityArray, 0)

    if similarityArray[similarityArray.argmax()] < similariyValue:
        return '#'
    else:
        return searchArray[similarityArray.argmax()]

def GetIndexOfMostSimilar():
    
    return '#'

def GetInput():
    try:
        userInput = input("> ")
        if userInput == '':
            return "#"
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        sys.exit()
    return userInput

#######################################################
# Main loop
#######################################################
while True:

    #Get input
    userInput = GetInput()
    
    #Check if input can be handled by aiml
    answer = kern.respond(userInput)
    if answer[0] != '#':

        #if answer.contains("Command-Describe"):
            
        #else:
        print(answer)
        
    else:

        mostSimilarDog = GetMostSimilar(userInput, breeds, 0.95)
        
        if mostSimilarDog != '#':
            print(breedInfo[breeds.index(mostSimilarDog)])
            
        else:

            #Check if they're naming a dog breed incorrectly
            mostSimilarDog = GetMostSimilar(userInput, breeds, 0.6)
            if mostSimilarDog != '#':
                print("Sorry, did you mean a " + mostSimilarDog + "? Would you like more information on this breed?")
                if (GetInput() in waysOfSayingYes):
                    print(breedInfo[breeds.index(mostSimilarDog)])
                    continue
                else:
                    print("Apologies I couldn't help")
                    continue

            #Check if they're describing a dog breed
            similarDescription = GetMostSimilar(userInput, breedInfo, 0.2)
            if similarDescription != '#':
                breedOfDescription = breeds[breedInfo.index(similarDescription)]
                print("Seems like you may be trying to describe a " + breedOfDescription + ". Would you like more information on this breed?")
                if (GetInput() in waysOfSayingYes):
                    print(similarDescription)
                    continue
                else:
                    print("Apologies I couldn't help")
                    continue
                    
            print("I did not get that, please try again.")
        
    

    
