﻿#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
lettuces => {}
cabbages => {}
mustards => {}
potatoes => {}
onions => {}
carrots => {}
beans => {}
peas => {}
field1 => f1
field2 => f2
field3 => f3
field4 => f4
be_in => {}
"""
folval = nltk.Valuation.fromstring(v)
grammar_file = 'simple-sem.fcfg'
objectCounter = 0


#######################################################
#  Initialise AIML agent
#######################################################
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="chatbot-aiml.xml")

#######################################################
# Welcome user
#######################################################
print("Welcome to the urban agriculture chat bot. Please feel free to ask questions about",
      "concepts and methods in making your garden a food production site,  Permaculture,",
      "Aquaponics, crops, the weather, or any plant images you might have.")
#######################################################
# Main loop
#######################################################
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 1:
            wpage = wiki_wiki.page(params[1])
            if wpage.exists():
                print(wpage.summary)
                print("Learn more at", wpage.canonicalurl)
            else:
                print("Sorry, I don't know what that is.")
        elif cmd == 2:
            succeeded = False
            api_url = r"http://api.openweathermap.org/data/2.5/weather?q="
            response = requests.get(api_url + params[1] + r"&units=metric&APPID="+APIkey)
            if response.status_code == 200:
                response_json = json.loads(response.content)
                if response_json:
                    t = response_json['main']['temp']
                    tmi = response_json['main']['temp_min']
                    tma = response_json['main']['temp_max']
                    hum = response_json['main']['humidity']
                    wsp = response_json['wind']['speed']
                    wdir = response_json['wind']['deg']
                    conditions = response_json['weather'][0]['description']
                    print("The temperature is", t, "°C, varying between", tmi, "and", tma, "at the moment, humidity is", hum, "%, wind speed ", wsp, "m/s,", conditions)
                    succeeded = True
            if not succeeded:
                print("Sorry, I could not resolve the location you gave me.")
        elif cmd == 7: # I will plant x in y
            o = 'o' + str(objectCounter)
            objectCounter += 1
            folval['o' + o] = o #insert constant
            if len(folval[params[1]]) == 1: #clean up if necessary
                if ('',) in folval[params[1]]:
                    folval[params[1]].clear()
            folval[params[1]].add((o,)) #insert type of plant information
            if len(folval["be_in"]) == 1: #clean up if necessary
                if ('',) in folval["be_in"]:
                    folval["be_in"].clear()
            folval["be_in"].add((o, folval[params[2]])) #insert location
        elif cmd == 8: #Are there any x in y
            g = nltk.Assignment(folval.domain)
            m = nltk.Model(folval.domain, folval)
            sent = 'some ' + params[1] + ' are_in ' + params[2]
            results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
            if results[2] == True:
                print("Yes.")
            else:
                print("No.")
        elif cmd == 9: # Are all x in y
            g = nltk.Assignment(folval.domain)
            m = nltk.Model(folval.domain, folval)
            sent = 'all ' + params[1] + ' are_in ' + params[2]
            results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
            if results[2] == True:
                print("Yes.")
            else:
                print("No.")
        elif cmd == 10: # Which plants are in ...
            g = nltk.Assignment(folval.domain)
            m = nltk.Model(folval.domain, folval)
            e = nltk.Expression.fromstring("be_in(x," + params[1] + ")")
            sat = m.satisfiers(e, "x", g)
            if len(sat) == 0:
                print("None.")
            else:
                #find satisfying objects in the valuation dictionary,
		#and print their type names
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
            print("I did not get that, please try again.")
    else:
        print(answer)
