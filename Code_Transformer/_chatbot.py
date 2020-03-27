#######################################################
# Imports
#######################################################
import aiml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import sys
import inflect
import nltk
import wikipediaapi
import os
import re
import tensorflow as tf
import tensorflow_datasets as tfds
import _ddqn_for_guessing_game
import gym
import random
import _transformer_for_chatbot

#######################################################
# Initialize wikipedia api
#######################################################
import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('en')
wikipediaapi.log.setLevel(level=wikipediaapi.logging.ERROR)


#######################################################
# Reasoning vars
#######################################################
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
locations = ["field1", "field2", "field3", "field4", "the_lake"]
actions = ["barking", "sitting", "walking", "lying_down"]
dogs = ["rosie", "bob", "dennis", "spark", "charlie", "max", "rover"]
canChase = dogs + ["bone"]
canBeOn = ["leash"]
canBePlaced = ["trees", "grass"] + canChase
folval = nltk.Valuation.fromstring(toyWorldString)
grammar_file = '_simple-sem.fcfg'
objectCounter = 0

#######################################################
# For singularization 
#######################################################
p = inflect.engine()

#######################################################
# Create a Kernel object & bootstrap to aiml file.
#######################################################
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="_chatbot-aiml.xml")

#######################################################
# Global vars
#######################################################
stopWords = ["the","is","an","a","at","and","i"]
waysOfSayingYes = ["yes", "y", "correct", "affirmative", "okay", "ok", "right", "of course", "by all means",
                   "sure", "indeed", "yea", "yeah", "yep", "yup", "certainly"]
breeds = []
breedInfo = []
sizes = []
NUM_BREEDS = None

transformer_model, START_TOKEN, END_TOKEN, MAX_LENGTH, tokenizer = _transformer_for_chatbot.get_model()

guessing_game_model = _ddqn_for_guessing_game.DDQNAgent(np.array([1]), gym.spaces.Discrete(3))

#######################################################
# ResetToyWorld
#######################################################
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
        
        print("Loading guessing game model weights...")
        guessing_game_model.load_weights('C:/Users/elliot/Documents/Github/ddqn_weights/ddqn_gameGuesser.h5')
        
        NUM_BREEDS = len(sizes)
        
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
        
    print(evaluate(search))
    return 0

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

def CheckCorrectInput(filter, toCheck):
    for item in filter:
        if item == toCheck:
            return True
    return False

#Set values/actions/positions
def SetFolValValues(data):
    global objectCounter
    if data[0] in folval:
        if data[2] in folval:    

            if data[1] == "be_in" and (CheckCorrectInput(locations, data[0]) or not CheckCorrectInput(locations, data[2])):
                print("Sorry, either " + data[2] + " is not a location, or " + data[0] + " is one.")
                return
                         
            if data[1] == "is" and (not CheckCorrectInput(actions, data[2]) or not CheckCorrectInput(dogs, data[0])):
                print("Sorry, either " + data[2] + " is not an action, or " + data[0] + " is not a dog.")
                return

            if data[1] == "is_chasing" and (not CheckCorrectInput(dogs, data[0]) or not CheckCorrectInput(canChase, data[2])):
                print("Sorry, either " + data[0] + " is not a dog, or a " + data[2] + " can't be chased.")
                return

            if data[1] == "is_on" and (not CheckCorrectInput(canBeOn, data[2]) or not CheckCorrectInput(dogs, data[0])):
                print("Sorry, either " + data[2] + " can't be applied to a dog, or " + data[0] + " is not a dog.")
                return

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

def DescribeToyWorld():
    print("The toy world contains: ")
    print("Locations: " + ', '.join(locations))    
    print("Dogs: " + ', '.join(dogs))
    print("Actions for dogs: " + ', '.join(actions))
    print("Chaseable objects: " + ', '.join(canChase))
    print("Placeable objects: " + ', '.join(canBePlaced))
    print("Try the command: Put rover in field1")
    

#######################################################
# Preprocess sentence ~ format sentence for processing in transformer model
#######################################################
def preprocess_sentence(sentence):
  sentence = sentence.lower().strip()
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
  sentence = sentence.strip()
  return sentence

#######################################################
# Predict ~ Return result of input sentence using transformer model
#######################################################
def evaluate(sentence):
  
  sentence = preprocess_sentence(sentence)
  sentence = tf.expand_dims(START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = transformer_model(inputs=[sentence, output], training=False)

    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    output = tf.concat([output, predicted_id], axis=-1)

  prediction = tf.squeeze(output, axis=0)
  predicted_sentence = tokenizer.decode(
    [i for i in prediction if i < tokenizer.vocab_size])
  
  return predicted_sentence
  
#######################################################
# Game_IsCorrectChoice ~ Returns weather or not answer 
# to guessing game is correct
#######################################################
def Game_IsCorrectChoice(choice, answer):
    choice = choice.lower()

    if choice == "small" or choice == "s":
        return (answer == "S")
    elif choice == "medium" or choice == "m":
        return (answer == "M")
    elif choice == "large" or choice == "L":
        return (answer == "L")
    else:
        return False
        
#######################################################
# Game_BotGuess ~ Returns bots guess to answer of guessing game
#######################################################
def Game_BotGuess(dogIndex):
    
    guess = guessing_game_model.act(np.reshape(np.array([dogIndex]), [1, 1]))
    if guess == 0:
        print("Guess : Small")
        return "Small"
    elif guess == 1:
        print("Guess : Medium")
        return "Medium"
    else:
        print("Guess : Large")
        return "Large"

#######################################################
# Game_MatchGameMainLoop 
#######################################################
def Game_MatchGameMainLoop(breeds, sizes, numOfBreeds, userIsPlaying):

    print("Aim of the game is to guess the size of a dog. (Small, Medium, Large). Type 'quit' to stop.")

    gamesCounter = 0
    correct = 0
    incorrect = 0
    while True:
        print()
    
        if gamesCounter == 10 and not userIsPlaying:
            break

        dogIndex = random.randrange(0, numOfBreeds)
        print("Guess the size of a " + breeds[dogIndex])
        
        playerInput = None
        if userIsPlaying:
            playerInput = GetInput()
        else:
            playerInput = Game_BotGuess(dogIndex)
            gamesCounter += 1
        
        if playerInput == "quit":
            print()
            break        
        elif Game_IsCorrectChoice(playerInput, sizes[dogIndex]):
            print("Correct!")
            correct+=1
        else:
            print("Incorrect!")
            incorrect+=1
    
    print("Correct: " + str(correct) + " Incorrect: " + str(incorrect))
    print("See you next time!")
    print()


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
    elif cmd == 10:
        DescribeToyWorld()  
    elif cmd == 11:
        FolValAnyMeet(data)
    elif cmd == 12:
        ResetToyWorld()
        print("Done.")
    elif cmd == 13:
        Game_MatchGameMainLoop(breeds, sizes, len(breeds), True)
    elif cmd == 14:
        Game_MatchGameMainLoop(breeds, sizes, len(breeds), False)
    elif cmd == 99:
        HandleUnknownInput(data[0])

#######################################################
# Main loop
#######################################################
def MainLoop():    
    print(("\nHi! I'm the dog breed information chatbot.\n - Try asking me a question about a specifc breed. \n - Ask me about groups of breeds(hounds, terriers, retrievers).\n - Try and describe a breed for me to guess. \n - Ask me to tell you a dog related joke.\n - Or ask me about the toy world.\n - Or say 'Play the guessing game'.\n - Or say 'Demonstrate the guessing game'.\n"))
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