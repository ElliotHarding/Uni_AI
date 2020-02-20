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
v = """
field1 => f1
field2 => f2
field3 => f3
field4 => f4
the_lake => the_lake
Rosie => Rosie
rover => rover
bob => bob
dennis => dennis
spark => spark
charlie => charlie
max => max
dog => {rosie, rover, bob, dennis, spark, charlie}
is_in => {}
is_on => {}
is_chasing => {}
is_below => {}
climbs => {}
lies => {}
walks => {}
barks => {}
sits => {}
sees => {}
bone => {}
"""
folval = nltk.Valuation.fromstring(v)
grammar_file = 'simple-sem.fcfg'
objectCounter = 0

#For singularization 
p = inflect.engine()

# Create a Kernel object & bootstrap to aiml file.
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="chatbot-aiml.xml")

#Global vars
stopWords = ["the","is","an","a","at","and","i"]
waysOfSayingYes = ["yes", "y", "correct", "affirmative", "okay", "ok", "right", "of course", "by all means",
                   "sure", "indeed", "yea", "yeah", "yep", "yup", "certainly"]
breeds = []
breedInfo = []
sizes = []

def ResetWorld():
    v = """
    field1 => f1
    field2 => f2
    field3 => f3
    field4 => f4
    the_lake => the_lake
    Rosie => Rosie
    rover => rover
    bob => bob
    dennis => dennis
    spark => spark
    charlie => charlie
    max => max
    dog => {rosie, rover, bob, dennis, spark, charlie}
    is_in => {}
    is_on => {}
    is_chasing => {}
    is_below => {}
    climbs => {}
    lies => {}
    walks => {}
    barks => {}
    sits => {}
    sees => {}
    bone => {}
    """
    folval = nltk.Valuation.fromstring(v)

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
# HandleAIMLCommand
#
#   Handles responses for AIML commands
#######################################################
def HandleAIMLCommand(cmd, data):
    global objectCounter
    global folval
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

    #Query
    elif cmd == 7:
        try:
            sent = " "
            sent = sent.join(data).lower()
            results = nltk.evaluate_sents([sent], grammar_file, nltk.Model(folval.domain, folval), nltk.Assignment(folval.domain))[0][0]
            if results[2] == True:
                print("Yes.")
            else:
                print("No.")
        except:
            print("Sorry, I don't know that.")

    #Set action --> x(PropN) z(TV in/below) y(PropN)
    elif cmd == 8:                
        if data[0] in folval:
            if data[2] in folval:                
                folval[data[1]].add((data[0], folval[data[2]]))
            else:
                print(data[2] + " does not exit in toy world.")
        else:
            print(data[0] + " does not exit in toy world.")
    
    #Set action --> x(PropN) z(TV below/chase/climbs/sees) y(-DET N)   
    elif cmd == 9:

        if data[0] in folval:
            if data[2] in folval:                

                #insert constant
                o = 'o' + str(objectCounter)
                objectCounter += 1
                folval['o' + o] = o

                #insert type of object information        
                folval[data[0]].add((o,))

                #insert location 
                folval[data[1]].add((o, folval[data[2]])) 
                
            else:
                print(data[2] + " does not exit in toy world.")
        else:
            print(data[0] + " does not exit in toy world.")

        

    #Are there any x in y    
    elif cmd == 88: 
        try:
            sent = 'some ' + data[1] + ' are_in ' + data[2]
            results = nltk.evaluate_sents([sent], grammar_file, nltk.Model(folval.domain, folval), nltk.Assignment(folval.domain))[0][0]
            if results[2] == True:
                print("Yes.")
            else:
                print("No.")
        except:                            
            print("No.")
            
    # Are all x in y      
    elif cmd == 99: 
        sent = 'all ' + data[1] + ' are_in ' + data[2]
        results = nltk.evaluate_sents([sent], grammar_file, nltk.Model(folval.domain, folval), nltk.Assignment(folval.domain))[0][0]
        if results[2] == True:
            print("Yes.")
        else:
            print("No.")

    # Which plants are in ...        
    elif cmd == 10: 
        sat = nltk.Model(folval.domain, folval).satisfiers(nltk.Expression.fromstring("be_in(x," + data[1] + ")"), "x", nltk.Assignment(folval.domain))
        if len(sat) == 0:
            print("None.")
        else:
            #find satisfying objects in the valuation dictionary, and print their type names
            sol = folval.values()
            for so in sat:
                for k, v in folval.items():
                    if len(v) > 0:
                        vl = list(v)
                        if len(vl[0]) == 1:
                            for i in vl:
                                if i[0] == so:
                                    print(k)
                                    break
    

    
    elif cmd == 99:
        if HandleUnknownInput(data[0]) == 0:
            print("I did not get that, please try again.")

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
        if answer[0] == '#':
            
            #Split answer into cmd & input
            params = answer[1:].split('$')
            HandleAIMLCommand(int(params[0]), params[1:])

        #Otherwise direct respond
        else:
            print(answer)


            
ReadFiles()
MainLoop()
    
