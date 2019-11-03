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
stopWords = ["the","is","an","a","at","and","i"]
waysOfSayingYes = ["yes", "y", "correct", "affirmative", "okay", "ok", "right", "of course", "by all means",
                   "sure", "indeed", "yea", "yeah", "yep", "yup", "certainly"]
breeds = []
breedInfo = []
sizes = []

#######################################################
# ReadFiles
#   
#   Reads breed-and-information file into global arrays
#   breeds & breedInfo
#######################################################
def ReadFiles():
    global breeds, breedInfo, sizes
    try:        
        breedAndInfoPairs = list(csv.reader(open('breed-and-information.csv', 'r')))
        
        breeds = [row[0] for row in breedAndInfoPairs]
        breedInfo = [row[1] for row in breedAndInfoPairs]
        sizes = [row[2] for row in breedAndInfoPairs]
        
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
# CheckSimilarDogs
#
#
#######################################################
def CheckSimilarDogs(userInput, arrayToCheck, similariyBound):

    dogsAndSimilarity = GetSimilarityArray(userInput, arrayToCheck)

    dogsToCheck = ""
    index = 0
    for i in dogsAndSimilarity:
        if i > similariyBound:
            dogsToCheck += breeds[index] + ", "
        index+=1

    arrayLen = len(dogsToCheck)
    dogsToCheck = dogsToCheck[0:arrayLen-2]

    if arrayLen > 0:
        print("You can ask me about any of these dogs: " + dogsToCheck)
        return 1
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
        HandleUnknownInput(dogName)
            
#######################################################
# HandleUnknownInput
#
#   Attempts to find figure out if user is asking for
#   information on a dog
#######################################################
def HandleUnknownInput(search):

    if CheckSimilarDogs(search, breeds, 0.3) == 1:
        return

    #Search may be plural?
    length = len(search)
    if search[length-1] == "s":
        if CheckSimilarDogs(search[0:length-1], breeds, 0.3) == 1:
            return

    if CheckSimilarDogs(search, breedInfo, 0.3) == 1:
        return
        
    print("It seems I couldn't find what you were looking for. Would you like me to wikipekida search '" + search + "'?")
    if (GetInput() in waysOfSayingYes):
        WikiSearch(dogName)
    else:
        print("Right.")

#######################################################
# PrintDogSize
#
#   Prints size of a dog
#######################################################
def PrintDogSize(dogName):
    iMostSimilarDog = GetIndexOfMostSimilar(dogName, breeds, 0.8)
    if iMostSimilarDog != -1:
        if sizes[iMostSimilarDog] == "L":
            print("A " + breeds[iMostSimilarDog] + " is a large sized dog.")
            return
        elif sizes[iMostSimilarDog] == "M":
            print("A " + breeds[iMostSimilarDog] + " is a medium sized dog")
            return
        elif sizes[iMostSimilarDog] == "S":
            print("A " + breeds[iMostSimilarDog] + " is a small sized dog")
            return
    print("Sorry I don't know the size of that dog.")

#######################################################
# ListSizedDogs
#
#   Prints list of dogs of specified size
#######################################################
def ListSizedDogs(size):
    breedList = ""
    index = 0
    for dog in breeds:
        if sizes[index] == size:
            breedList += dog + ", "
        index+=1

    #Get rid of last ", "
    breedList = breedList[0:len(breedList)-2]
    print("Here's the dogs I found: " + breedList)

#######################################################
# IsCrossBreed
#
#   Returns if a dog at index iDog in breedInfo is a cross breed or not
#######################################################
def IsCrossBreed(iDog):
    dogDescription = breedInfo[iDog]
    if "mixed breed" in dogDescription or "a cross" in dogDescription:
        return 1
    else:
        return 0

#######################################################
# PrintCrossBreed
#
#   Prints out if a dog is a cross breed or not
#######################################################
def PrintCrossBreed(dog):
    print(dog)
    iMostSimilarDog = GetIndexOfMostSimilar(dog, breeds, 0.5)
    if iMostSimilarDog != -1:        
        if IsCrossBreed(iMostSimilarDog):
            print("A " + breeds[iMostSimilarDog] + " is in fact a cross breed.")
        else:
            print("A " + breeds[iMostSimilarDog] + " is in fact a pure breed.")
        return
            
    if CheckSimilarDogs(dog, breeds, 0.3) == 1:
        return
    else:
        print("It seems I couldn't find what you were looking for. Would you like me to wikipekida search '" + dog + "'?")
    
 
#######################################################
# PrintCrossBreeds
#
#   Prints out all cross breeds
#######################################################
def PrintCrossBreeds():
    breedList = ""
    index = 0
    for dog in breeds:
        if IsCrossBreed(index):
            breedList += dog + ", "
        index+=1

    #Get rid of last ", "
    breedList = breedList[0:len(breedList)-2]
    print("Here are all the cross breeds: " + breedList)

#######################################################
# Main loop
#######################################################
def MainLoop():
    
    print("Hi! I'm the dog breed information chatbot.\nTry asking me a question about a specifc breed, ask me about groups of breeds\n(hounds, terriers, retrievers), try and describe a breed,\nor just make general dog related chit-chat!")
    
    while True: 

        #Get input
        userInput = GetInput()

        #Get response from aiml
        answer = kern.respond(userInput)
        print(answer)
        if answer[0] == '#':

            #Split answer into cmd & input
            params = answer[1:].split('$')
            cmd = int(params[0])

            if cmd == 0:
                Exit()
                
            elif cmd == 1:
                DescribeDog(params[1])
                continue

            elif cmd == 2:
                WikiSearch(params[1])
                continue

            elif cmd == 3:
                print(params[1])
                PrintCrossBreed(params[1])
                continue

            elif cmd == 4:
                PrintCrossBreeds()
                continue

            elif cmd == 5:
                PrintDogSize(params[1])
                continue

            elif cmd == 6:
                ListSizedDogs(params[1])
                continue
            
            elif cmd == 99:
                if HandleUnknownInput(params[1]) == 0:
                    print("I did not get that, please try again.")
                continue
  
        else:
            print(answer)
            
            
ReadFiles()
MainLoop()
    
