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

#Initialize wikipedia api
import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('en')
wikipediaapi.log.setLevel(level=wikipediaapi.logging.ERROR)

#Global vars
stopWords = ["the","is","an","a","at","and"]
waysOfSayingYes = ["yes", "y", "correct", "affirmative", "okay", "ok", "right", "of course", "by all means",
                   "sure", "indeed", "yea", "yeah", "yep", "yup", "certainly"]
breeds = []
breedInfo = []

#######################################################
# ReadFiles
#   
#   Reads breed-and-information file into global arrays
#   breeds & breedInfo
#######################################################
def ReadFiles():
    global breeds, breedInfo
    try:        
        breedAndInfoPairs = list(csv.reader(open('breed-and-information.csv', 'r')))
        
        breeds = [row[0] for row in breedAndInfoPairs]
        breedInfo = [row[1] for row in breedAndInfoPairs]
        
    except (IOError) as e:
        print("Error occured opening one of the files! " + e.strerror)
        sys.exit()

#######################################################
# GetSimilarityArray
#   
#######################################################
def GetSimilarityArray(string, searchArray):
    array = [string] + searchArray
    tfidf = TfidfVectorizer(stop_words=stopWords).fit_transform(array)
    similarityArray = cosine_similarity(tfidf[0:1], tfidf)
    similarityArray = np.delete(similarityArray, 0)

    return similarityArray

#######################################################
# GetIndexOfMostSimilar
#   Checks input list for most similar string to input string
#   Returns -1 if no strings over similarityBound
#######################################################
def GetIndexOfMostSimilar(string, searchArray, similariyBound=0):
    similarityArray = GetSimilarityArray(string, searchArray)
    if similarityArray[similarityArray.argmax()] < similariyBound:
        return -1
    else:
        return similarityArray.argmax()

#######################################################
# Exit
#######################################################
def Exit():
    print("Bye!")
    sys.exit()

#######################################################
# GetInput
#   Handles & returns user input
#######################################################
def GetInput():
    try:
        userInput = input("> ")
        if userInput == '':
            return "-"
    except (KeyboardInterrupt, EOFError) as e:
        Exit()
    return userInput

#######################################################
# CheckDescribeDog
#   Checks if user maybe trying to describe a dog,
#   uses passed array to check
#
#
#######################################################
def CheckDescribeDog(userInput, arrayToCheck, similariyBound):
    iDog = GetIndexOfMostSimilar(userInput, arrayToCheck, similariyBound)
    if iDog != -1:
        print("Seems like you may be trying to describe a " + breeds[iDog] + ". Would you like information on this breed?")
        if (GetInput() in waysOfSayingYes):
            print(breedInfo[iDog])
            return 1
        else:
            print("Right.")
            return 2
    return 0

#######################################################
# CheckSimilarDogNames
#
#
#######################################################
def CheckSimilarDogNames(userInput, similariyBound):

    dogsAndSimilarity = GetSimilarityArray(userInput, breeds)

    dogsToCheck = ""
    index = 0
    for i in dogsAndSimilarity:
        if i > similariyBound:
            dogsToCheck += breeds[index] + ", "
        index+=1

    arrayLen = len(dogsToCheck)
    dogsToCheck = dogsToCheck[0:arrayLen-2]

    if arrayLen > 0:
        print("Which of these dogs would you like to know more about? " + dogsToCheck)
        iMostSimilarDog = GetIndexOfMostSimilar(GetInput(), breeds, 0.7)        
        if iMostSimilarDog != -1:
            print(breedInfo[iMostSimilarDog])
            return 1
        print("Sorry it's not in the list!")
    return 0

#######################################################
# WikiSearch
#######################################################
def WikiSearch(search):    
    wpage = wiki_wiki.page(search)
    if wpage.exists():
        print(wpage.summary)
        print("Learn more at", wpage.canonicalurl)
    else:
        print("Sorry, I don't know what that is.")
            
#######################################################
# DescribeDog
# describes a dog
#######################################################
def DescribeDog(dogName):

    iMostSimilarDog = GetIndexOfMostSimilar(dogName, breeds, 0.8)        
    if iMostSimilarDog != -1:
        print(breedInfo[iMostSimilarDog])
        
    else:
        #Handle case if user makes searched dog plural
        length = len(dogName) - 1
        if dogName[length].lower() == 's':
            DescribeDog(dogName[0:length])
            return

        if CheckSimilarDogNames(dogName, 0.3) == 1:
            return

        print("It seems I couldn't find what you were looking for. Would you like me to wikipekida search '" + dogName + "'?")
        if (GetInput() in waysOfSayingYes):
            WikiSearch(dogName)
        else:
             print("Right.")
            
#######################################################
# HandleUnknownInput
#
#   Attempts to find figure out if user is asking for
#   information on a dog
#######################################################
def HandleUnknownInput(search):

    ret = CheckDescribeDog(search, breeds, 0.2)
    if ret == 1:
        return 1 
    
    elif ret == 2:
        ret = CheckDescribeDog(search, breedInfo, 0.1)
        if ret == 1:
            return 1
        elif ret == 0:
            #Handle plural input, only after inital search,
            #since dog name may just has an s at the end
            length = len(search) - 1
            if search[length].lower() == 's':
                return HandleUnknownInput(search[0:length])
        return ret
    else:
        #Handle plural input, only after inital search,
        #since dog name may just has an s at the end
        length = len(search) - 1
        if search[length].lower() == 's':
            return HandleUnknownInput(search[0:length])
        return ret

#######################################################
# Main loop
#######################################################
def MainLoop():
    
    previousWasQuestion = 0

    prompt = ""
    print("Hi! I'm the dog breed information chatbot.\nTry asking me a question about a specifc breed, ask me about groups of breeds\n(hounds, terriers, retrievers), try and describe a breed,\nor just make general dog related chit-chat!")
    
    while True: 

        #Get input
        userInput = GetInput()

        if previousWasQuestion:
            previousWasQuestion = 0
            print("Nice!")
            continue

        #Get response from aiml
        answer = kern.respond(userInput)
        if answer[0] == '#':

            #Split answer into cmd & input
            params = answer[1:].split('$')
            cmd = int(params[0])

            if params[1] == "":
                print("I did not get that, please try again.")
                continue

            if cmd == 0:
                Exit()
                
            elif cmd == 1:
                DescribeDog(params[1])
                continue

            elif cmd == 2:
                WikiSearch(params[1])
                continue
                    
            elif cmd == 99:
                if HandleUnknownInput(params[1]) == 0:
                    print("I did not get that, please try again.")
                continue
  
        else:
            print(answer)

        if '?' in answer:
            previousWasQuestion = 1
            
            
ReadFiles()
MainLoop()
    
