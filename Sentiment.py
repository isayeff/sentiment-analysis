#!/usr/bin/env python
import re, random

PRINT_ERRORS=0

#------------- Function Definitions ---------------------

def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

    #reading pre-labeled input and splitting into lines
    posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    posSentencesNokia = open('nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())
 
    posDictionary = open('positive-words.txt', 'r', encoding="ISO-8859-1")
    posWordList = []
    
    for line in posDictionary:
        word = line.strip()
        if word and not word.startswith(';'):
            posWordList.append(word)

    negDictionary = open('negative-words.txt', 'r', encoding="ISO-8859-1")
    negWordList = []
    
    for line in negDictionary:
        word = line.strip()
        if word and not word.startswith(';'):
            negWordList.append(word)
                
    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    # Create Training and Test Datsets:
    # We want to test on sentences we haven't trained on, 
    # to see how well the model generalses to previously unseen sentences

    # create 90-10 split of training and test data from movie reviews, with sentiment labels    
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    # create Nokia Datset:
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"

#----------------------------End of data initialisation ----------------#

# calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    posFeatures = [] # [] initialises a list [array]
    negFeatures = [] 
    freqPositive = {} # {} initialises a dictionary [hash function]
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0

    # iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.items():
        wordList = re.findall(r"[\w']+", sentence)
        
        for word in wordList: # calculate over unigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment=="positive" :
                posWordsTot += 1 # keeps count of total words in positive class

                # keep count of each word in positive context
                if not (word in freqPositive):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1 # keeps count of total words in negative class
                
                # keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        # do some smoothing so that minimum count of a word is 1
        if not (word in freqNegative):
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 

#---------------------------End Training ----------------------------------

# implement naive bayes algorithm
# INPUTS:
#   sentencesTest is a dictonary with sentences associated with sentiment 
#   dataName is a string (used only for printing output)
#   pWordPos is dictionary storing p(word|positive) for each word
#      i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#   pWordNeg is dictionary storing p(word|negative) for each word
#   pWord is dictionary storing p(word)
#   pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord,pPos):
    
    pNeg=1-pPos

    # These variables will store results
    total=0
    correct=0
    totalpos=0
    totalpospred=0
    totalneg=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    # for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)#collect all words

        pPosW=pPos
        pNegW=pNeg

        for word in wordList: # calculate over unigrams
            if word in pWord:
                if pWord[word]>0.00000001:
                    pPosW *=pWordPos[word]
                    pNegW *=pWordNeg[word]

        prob=0;            
        if pPosW+pNegW >0:
            prob=pPosW/float(pPosW+pNegW)


        total+=1
        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %prob + sentence)
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %prob + sentence)
    
    # Calculate accuracy, precision, recall, F1 score
    accuracy = correct / total
    precision_pos = correctpos / totalpospred if totalpospred != 0 else 0
    recall_pos = correctpos / totalpos if totalpos != 0 else 0
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if precision_pos + recall_pos != 0 else 0

    precision_neg = correctneg / totalnegpred if totalnegpred != 0 else 0
    recall_neg = correctneg / totalneg if totalneg != 0 else 0
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if precision_neg + recall_neg != 0 else 0

    # Print out the evaluation metrics
    print(f"\n{'='*50}")
    print(f"{dataName}")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Positive): {precision_pos:.4f}")
    print(f"Precision (Negative): {precision_neg:.4f}")
    print(f"Recall (Positive): {recall_pos:.4f}")
    print(f"Recall (Negative): {recall_neg:.4f}")
    print(f"F1 Score (Positive): {f1_pos:.4f}")
    print(f"F1 Score (Negative): {f1_neg:.4f}")
 
 

# This is a simple classifier that uses a sentiment dictionary to classify 
# a sentence. For each word in the sentence, if the word is in the positive 
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1. 
# If the final score is above a threshold, it classifies as "Positive", 
# otherwise as "Negative"
def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):

    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            w = word.lower()
            if w in sentimentDictionary:
               score+=sentimentDictionary[w]
 
        total += 1
        if sentiment == "positive":
            totalpos += 1
            if score >= threshold:
                correct += 1
                correctpos += 1
                totalpospred += 1
            else:
                totalnegpred += 1
                if PRINT_ERRORS:
                    print(f"ERROR (positive classified as negative): {sentence}")
        else:
            totalneg += 1
            if score < threshold:
                correct += 1
                correctneg += 1
                totalnegpred += 1
            else:
                totalpospred += 1
                if PRINT_ERRORS:
                    print(f"ERROR (negative classified as positive): {sentence}")
    
    print(f"\n{'='*50}")
    print(f"{dataName}")
    print(f"{'='*50}")
    
    accuracy = correct / total
    precision_pos = correctpos / totalpospred if totalpospred != 0 else 0
    recall_pos = correctpos / totalpos if totalpos != 0 else 0
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if precision_pos + recall_pos != 0 else 0

    precision_neg = correctneg / totalnegpred if totalnegpred != 0 else 0
    recall_neg = correctneg / totalneg if totalneg != 0 else 0
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if precision_neg + recall_neg != 0 else 0

    # Print out the evaluation metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Positive): {precision_pos:.4f}")
    print(f"Precision (Negative): {precision_neg:.4f}")
    print(f"Recall (Positive): {recall_pos:.4f}")
    print(f"Recall (Negative): {recall_neg:.4f}")
    print(f"F1 Score (Positive): {f1_pos:.4f}")
    print(f"F1 Score (Negative): {f1_neg:.4f}")
 
    

# Print out n most useful predictors
def mostUseful(pWordPos, pWordNeg, pWord, n):
    predictPower={}
    for word in pWord:
        if pWordNeg[word]<0.0000001:
            predictPower[word] = 1000000000
        else:
            predictPower[word]=pWordPos[word] / (pWordPos[word] + pWordNeg[word])
            
    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower)-n:]
    print(f'\n{"=" * 50}')
    print("MOST USEFUL WORDS")
    print(f'{"=" * 50}')
    print ("NEGATIVE:")
    print (head)
    print ("\nPOSITIVE:")
    print (tail)
    
# STEP 5 - Improving the Rule-Based System

def testDictionaryWithNegation(sentencesTest, dataName, sentimentDictionary, threshold):
   
    print(f"\n{'='*50}")
    print(f"{dataName}")
    print(f"{'='*50}")

    total = 0
    correct = 0
    totalpos = 0
    totalneg = 0
    totalpospred = 0
    totalnegpred = 0
    correctpos = 0
    correctneg = 0

    # Simplified modifier sets - only the most common ones
    intensifiers = {'very', 'extremely', 'really', 'so', 'absolutely', 'totally', 'incredibly'}
    diminishers = {'slightly', 'somewhat', 'fairly', 'rather', 'quite', 'pretty'}
    negation_words = {'not', 'no', "n't", 'never', 'neither', 'nobody', 'nothing', 'nowhere', 'none'}

    for sentence, sentiment in sentencesTest.items():
        
        words = re.findall(r"[\w']+", sentence)
        score = 0
        
        i = 0
        while i < len(words):
            word = words[i]
            word_lower = word.lower()
            
            # Check if current word is a negation
            is_negated = False
            if word_lower in negation_words or word.endswith("n't"):
                is_negated = True
                i += 1
                if i >= len(words):
                    break
                word = words[i]
                word_lower = word.lower()
            
            # Check if current word is an intensifier
            intensifier_multiplier = 1.0
            if word_lower in intensifiers:
                intensifier_multiplier = 1.3 
                i += 1
                if i >= len(words):
                    break
                word = words[i]
                word_lower = word.lower()
            
            # Check if current word is a diminisher  
            elif word_lower in diminishers:
                intensifier_multiplier = 0.7  
                i += 1
                if i >= len(words):
                    break
                word = words[i]
                word_lower = word.lower()
            
            # Now process the actual sentiment word
            if word_lower in sentimentDictionary:
                word_score = sentimentDictionary[word_lower]
               
                word_score *= intensifier_multiplier
                
                if is_negated:
                    word_score = -word_score
                
                score += word_score
            
            i += 1

        # Classification
        total += 1
        if sentiment == "positive":
            totalpos += 1
            if score >= threshold:
                correct += 1
                correctpos += 1
                totalpospred += 1
            else:
                totalnegpred += 1
        else:
            totalneg += 1
            if score < threshold:
                correct += 1
                correctneg += 1
                totalnegpred += 1
            else:
                totalpospred += 1

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    precision_pos = correctpos / totalpospred if totalpospred != 0 else 0
    recall_pos = correctpos / totalpos if totalpos != 0 else 0
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if precision_pos + recall_pos != 0 else 0

    precision_neg = correctneg / totalnegpred if totalnegpred != 0 else 0
    recall_neg = correctneg / totalneg if totalneg != 0 else 0
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if precision_neg + recall_neg != 0 else 0

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Positive): {precision_pos:.4f}")
    print(f"Precision (Negative): {precision_neg:.4f}")
    print(f"Recall (Positive): {recall_pos:.4f}")
    print(f"Recall (Negative): {recall_neg:.4f}")
    print(f"F1 Score (Positive): {f1_pos:.4f}")
    print(f"F1 Score (Negative): {f1_neg:.4f}")


#---------- Main Script --------------------------


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

# build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

# run naive bayes classifier on datasets
testBayes(sentencesTrain,  "Films (Train Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesTest, "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord, 0.5)
testBayes(sentencesNokia, "Nokia   (All Data,  Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.7)



# run sentiment dictionary based classifier on datasets
testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, 1)
testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 1)
testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 1)


# Run dictionary-based classification with negation handling
testDictionaryWithNegation(sentencesTrain, "Films (Train Data, Rule-Based with Negation)", sentimentDictionary, 1)
testDictionaryWithNegation(sentencesTest, "Films (Test Data, Rule-Based with Negation)", sentimentDictionary, 1)
testDictionaryWithNegation(sentencesNokia, "Nokia (All Data, Rule-Based with Negation)", sentimentDictionary, 1)



# print most useful words
mostUseful(pWordPos, pWordNeg, pWord, 100)
