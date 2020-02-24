#Imports
import aiml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import sys
import inflect
import nltk
import wikipediaapi

#Initialize wikipedia api
import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('en')
wikipediaapi.log.setLevel(level=wikipediaapi.logging.ERROR)

#Reasoning
toyWorldString = """
field1 => f1
field2 => f2
field3 => f3
field4 => f4
leash => leash
bone => bone
the_lake => the_lake
barking => barking
walking => walking
sitting => sitting
lying_down => lying_down
rosie => rosie
bob => bob
dennis => dennis
spark => spark
charlie => charlie
max => max
rover => rover
dogs => {rosie, rover, bob, dennis, spark, charlie}
be_in => {}
trees => {}
grass => {}
is => {}
is_on => {}
is_chasing => {}
"""
folval = nltk.Valuation.fromstring(toyWorldString)
grammar_file = '_simple-sem.fcfg'
objectCounter = 0

#For singularization 
p = inflect.engine()

# Create a Kernel object & bootstrap to aiml file.
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="_chatbot-aiml.xml")

#Global vars
stopWords = ["the","is","an","a","at","and","i"]
waysOfSayingYes = ["yes", "y", "correct", "affirmative", "okay", "ok", "right", "of course", "by all means",
                   "sure", "indeed", "yea", "yeah", "yep", "yup", "certainly"]
breeds = []
breedInfo = []
sizes = []

def ResetToyWorld():
    global folval
    folval = nltk.Valuation.fromstring(toyWorldString)

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
        WikiSearch(search)
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

def ClearEmptyFolvalSlot(slot):
    if len(folval[slot]) == 1: 
        if ('',) in folval[slot]:
            folval[slot].clear()

def GetMatchingFolvalValues(linkWord, searchWord):
    return nltk.Model(folval.domain, folval).satisfiers(nltk.Expression.fromstring(linkWord+"(x," + searchWord + ")"), "x", nltk.Assignment(folval.domain))

#Set values/actions/positions
def SetFolValValues(data):
    global objectCounter
    if data[0] in folval:
        if data[2] in folval:

            o = data[0]
            
            #Can't be in same place or doing two actions at once
            for item in folval[data[1]]:
                if data[0] in item:
                    if data[1] == "be_in":
                        print(data[0] + " has moved to " + data[2])
                    elif data[1] == "is":                    
                        print(data[0] + " is now " + data[2])                     
                    folval[data[1]].remove(item)
                    break 

            if data[0] == "trees" or data[0] == "grass":              
                o = 'o' + str(objectCounter)
                objectCounter += 1
                folval['o' + o] = o
                ClearEmptyFolvalSlot(data[0])
                folval[data[0]].add((o,))                  

            ClearEmptyFolvalSlot(data[1])
            folval[data[1]].add((o, folval[data[2]]))     

            print("Done.")
        else:
            print(data[2] + " does not exist in toy world.")
    else:
        print(data[0] + " does not exist in toy world.")

#Yes/no queries
def FolValYesNoQueries(data):
    try:
        sent = " "
        results = nltk.evaluate_sents([sent.join(data).lower()], grammar_file, nltk.Model(folval.domain, folval), nltk.Assignment(folval.domain))[0][0]
        if results[2] == True:
            print("Yes.")
        else:
            print("No.")
    except:
        print("Sorry, I don't know that.")

#Query ~ List all values which meet x
def FolValListMatches(data):
    try:
        sat = GetMatchingFolvalValues(data[0], data[1])
        if len(sat) == 0:
            print(data[2])
        else:   
            for so in sat:
                if any(char.isdigit() for char in so) == True:                
                    for k, v in folval.items():
                        if len(v) > 0:
                            vl = list(v)
                            if len(vl[0]) == 1:
                                for i in vl:
                                    if i[0] == so:
                                        print(k)
                                        break
                else:
                    print(so)
    except:
        print("Sorry can't help with that")

#Query ~ do any values meet x
def FolValAnyMeet(data):
    if data[1] == "lying down":
        data[1] = "lying_down"

    if data[1] in folval:
        if len(GetMatchingFolvalValues(data[0], data[1])) == 0:
            print("No.")
        else:    
            print("yes.")
    else:
        print("That action is not in the toy world.")   

#######################################################
# HandleAIMLCommand
#
#   Handles responses for AIML commands
#######################################################
def HandleAIMLCommand(cmd, data):
    
    if cmd == 0:
        Exit()                
    elif cmd == 1:
        DescribeDog(data[0])
    elif cmd == 2:
        WikiSearch(data[0])
    elif cmd == 3:
        PrintCrossBreed(data[0])
    elif cmd == 4:
        PrintCrossBreeds()
    elif cmd == 5:
        PrintDogSize(data[0])
    elif cmd == 6:
        ListSizedDogs(data[0])
    elif cmd == 7: 
        SetFolValValues(data)
    elif cmd == 8:
        FolValYesNoQueries(data)
    elif cmd == 9:
        FolValListMatches(data)    
    elif cmd == 11:
        FolValAnyMeet(data)
    elif cmd == 12:
        ResetToyWorld()
        print("Done.")
    elif cmd == 99:
        if HandleUnknownInput(data[0]) == 0:
            print("I did not get that, please try again.")
    print(folval)

#######################################################
# Main loop
#######################################################
def MainLoop():    
    print(("\nHi! I'm the dog breed information chatbot.\n - Try asking me a question about a specifc breed. \n - Ask me about groups of breeds(hounds, terriers, retrievers).\n - Try and describe a breed for me to guess. \n - Or ask me to tell you a dog related joke.\n"))
    while True: 

        #Get input
        userInput = GetInput()

        #Get response from aiml
        answer = kern.respond(userInput)

        #Check if command response
        if userInput != "-" and answer != "" and answer[0] == '#':
            
            #Split answer into cmd & input
            params = answer[1:].split('$')
            HandleAIMLCommand(int(params[0]), params[1:])

        #Otherwise direct respond
        else:
            print(answer)
            
ReadFiles()
MainLoop()