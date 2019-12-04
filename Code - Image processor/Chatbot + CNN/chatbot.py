#Imports
import aiml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import sys
import inflect
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt
import pandas as pd
import cv2

#For singularization 
p = inflect.engine()

# Create a Kernel object & bootstrap to aiml file.
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="chatbot-aiml.xml")

#Initialize wikipedia api
import wikipediaapi #pip install Wikipedia-API
wiki_wiki = wikipediaapi.Wikipedia('en')
wikipediaapi.log.setLevel(level=wikipediaapi.logging.ERROR)

#Global vars
stopWords = ["the","is","an","a","at","and","i"]
waysOfSayingYes = ["yes", "y", "correct", "affirmative", "okay", "ok", "right", "of course", "by all means",
                   "sure", "indeed", "yea", "yeah", "yep", "yup", "certainly"]
imageEndingTypes = [".img",".jpeg",".jpg",".png"]

breeds = []
breedInfo = []
sizes = []

dogBreedClassifier = 0
IMG_SIZE = 244
classifierBreeds = []

#######################################################
# ReadFiles
#   
#   Reads breed-and-information file into global arrays
#   breeds & breedInfo
#######################################################
def ReadFiles():
    global breeds, breedInfo, sizes, classifierBreeds, dogBreedClassifier
    try:        
        breedAndInfoPairs = list(csv.reader(open('breed-and-information.csv', 'r')))
        
        breeds = [row[0] for row in breedAndInfoPairs]
        breedInfo = [row[1] for row in breedAndInfoPairs]
        sizes = [row[2] for row in breedAndInfoPairs]
        
        classifierBreedsCsv = list(csv.reader(open('cnn-breed-names.csv', 'r')))
        classifierBreeds = [row[0] for row in classifierBreedsCsv]

        dogBreedClassifier = tf.keras.models.load_model('E:\\dog-cnn.h5')
        
    except (IOError) as e:
        print("Error occured opening one of the files! ")
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

    if TrailImagePredictionRequest(data) == 1:
        return

    if CheckSimilarDogs(search, breeds, 0.3) == 1:
        return

    #If it's a single word, just quick check if its plural
    if not " " in search:
        singular = p.singular_noun(search)
        if singular:            
            if CheckSimilarDogs(p.singular_noun(search), breeds, 0.3) == 1:
                return

    if CheckSimilarDogs(search, breedInfo, 0.3) == 1:
        return

    #No point in doing a wiki search if related to bot
    if " you" in search or "you " in search or " me" in search or "me " in search:
        return 0
        
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
# PredictDogImage
#
#   Takes an image path and runs the corresponding image to that path
#   on the pre-trained CNN
#######################################################
def PredictDogImage(imagePath):
    print(imagePath)
    img = ""
    try:    
        img = image.load_img(imagePath, target_size=(IMG_SIZE, IMG_SIZE))
        print(img)
    except (Exception) as e:
        print("Sorry! I can't seem to find that image")
        return

    raw_image = cv2.imread(imagePath)
    scaled_image = cv2.resize(raw_image, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    image_array = scaled_image.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    prediction_list = dogBreedClassifier.predict(image_array)
    
    print(prediction_list)
    result=np.argmax(prediction_list,axis=1)

    try:    
        print("It should be a " + classifierBreeds[result[0]])      
    except (Exception) as e:
        print("Unknown dog!")

#######################################################
# TrailImagePredictionRequest
#
#   Searches through text for image types, if so finds the specified
#   image & calls PredictDogImage on it
#######################################################
def TrailImagePredictionRequest(string):
    for imageTypeEnding in imageEndingTypes:
        if imageTypeEnding in string:

            #Get the name that goes infront of the imgage type ending
            startIndex=string.find(imageTypeEnding)
            endIndex=startIndex
            while startIndex > -1:
                startIndex-=1
                if string[startIndex] == " " or string[startIndex] == "\\" or string[startIndex] == "/":
                    break

            if startIndex < endIndex:
                PredictDogImage(string[startIndex+1:endIndex]+imageTypeEnding)
            else:
                print("Sorry! I can't seem to find that image")
            return 1
    return 0
    

#######################################################
# HandleAIMLCommand
#
#   Handles responses for AIML commands
#######################################################
def HandleAIMLCommand(cmd, data):
    if cmd == 0:
        Exit()                
    elif cmd == 1:
        DescribeDog(data)
    elif cmd == 2:
        WikiSearch(data)
    elif cmd == 3:
        PrintCrossBreed(data)
    elif cmd == 4:
        PrintCrossBreeds()
    elif cmd == 5:
        PrintDogSize(data)
    elif cmd == 6:
        ListSizedDogs(data)
    elif cmd == 7:
        if TrailImagePredictionRequest(data) == 0:
            print("I did not get that, please try again.")
    elif cmd == 99:
        if HandleUnknownInput(data) == 0:
            print("I did not get that, please try again.")

#######################################################
# Main loop
#######################################################
def MainLoop():    
    print(("\nHi! I'm the dog breed information chatbot.\n - Try asking me a question about a specifc breed. \n - Ask me to predict the breed of a dog from an image. \n - Ask me about groups of breeds(hounds, terriers, retrievers).\n - Try and describe a breed for me to guess. \n - Or ask me to tell you a dog related joke.\n"))
 
    while True: 

        #Get input
        userInput = GetInput()

        #Get response from aiml & error check
        answer = kern.respond(userInput)
        if len(answer) == 0:
            continue

        answer = answer.replace("  #99$", ".")
        print(answer)

        #Check if command response
        if answer[0] == '#':
            
            #Split answer into cmd & input
            params = answer[1:].split('$')
            HandleAIMLCommand(int(params[0]), params[1])

        #Otherwise direct respond
        else:
            print(answer)
            
            
ReadFiles()
MainLoop()
    
