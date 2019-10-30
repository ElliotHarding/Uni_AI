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

#Global vars
stopWords = ["the","is","an","a","at","and"]
waysOfSayingYes = ["yes", "y", "correct", "affirmative", "okay", "ok", "right", "of course", "by all means",
                   "sure", "indeed", "yea", "yeah", "yep", "yup", "certainly"]
questions = []
answers = []
breeds = []
breedInfo = []

def ReadFiles():
    global questions, answers, breeds, breedInfo
    try:
        questionAnswerPairs = list(csv.reader(open('question-and-answer-pairs.csv', 'r')))
        breedAndInfoPairs = list(csv.reader(open('breed-and-info-pairs.csv', 'r')))
        
        questions = [row[0] for row in questionAnswerPairs]
        answers = [row[1] for row in questionAnswerPairs]
        breeds = [row[0] for row in breedAndInfoPairs]
        breedInfo = [row[1] for row in breedAndInfoPairs]
        
    except (IOError) as e:
        print("Error occured opening one of the files! " + e.message)
        sys.exit()

# GetMostSimilar : Checks input list for most similar string to input string
#   returns # if no string found
def GetIndexOfMostSimilar(string, searchArray, similariyBound=0):
    array = [string] + searchArray
    tfidf = TfidfVectorizer(stop_words=stopWords).fit_transform(array)
    similarityArray = cosine_similarity(tfidf[0:1], tfidf)
    similarityArray = np.delete(similarityArray, 0)

    if similarityArray[similarityArray.argmax()] < similariyBound:
        return -1
    else:
        return similarityArray.argmax()

def Exit():
    print("Bye!")
    sys.exit()

def GetInput():
    try:
        userInput = input("> ")
        if userInput == '':
            return "-"
    except (KeyboardInterrupt, EOFError) as e:
        Exit()
    return userInput

def CheckDescribeDog(userInput, arrayToCheck, similariyBound):
    iDog = GetIndexOfMostSimilar(userInput, arrayToCheck, similariyBound)
    if iDog != -1:
        print("Seems like you may be trying to describe a " + breeds[iDog] + ". Would you like information on this breed?")
        if (GetInput() in waysOfSayingYes):
            print(breedInfo[iDog])
            return 1
        else:
            print("Apologies. ")
            return 2
    return 0

def DescribeDog(dogName):
    iMostSimilarDog = GetIndexOfMostSimilar(dogName, breeds, 0.95)        
    if iMostSimilarDog != -1:
        print(breedInfo[iMostSimilarDog])
    else:
        print("Apologies, I don't have any info on that breed.")        

def HandleUnknownInput(userInput):

    ret = CheckDescribeDog(userInput, breeds, 0.6)
    if ret == 1:
        return
    
    elif ret == 2:
        ret = CheckDescribeDog(userInput, breedInfo, 0.4)
        if ret == 1:
            return
        elif ret == 0:
            print("I did not get that, please try again.")
    else:
        print("I did not get that, please try again.")

#######################################################
# Main loop
#######################################################
def MainLoop():
    
    previousWasQuestion = 0

    prompt = "Try asking me a question about a specifc breed, try and describe a breed, or just make general dog related chit-chat!"
    print("Hi! I'm the dog breed information chatbot. " + prompt)
    
    while True: 

        #Get input
        userInput = GetInput()

        if previousWasQuestion:
            previousWasQuestion = 0
            print("Nice! " + prompt)
            continue
        
        #Check if input can be handled by aiml
        answer = kern.respond(userInput)
        if answer[0] == '#':

            #Split answer into cmd & input
            params = answer[1:].split('$')
            cmd = int(params[0])

            if cmd == 0:
                Exit()
                
            elif cmd == 1:
                DescribeDog(params[1])
                continue
                    
            elif cmd == 99:
                HandleUnknownInput(params[1])
                continue
  
        else:
            print(answer)

        if '?' in answer:
            previousWasQuestion = 1
            
            
ReadFiles()
MainLoop()
    
