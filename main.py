import math
import os
import pickle
import nltk
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
import heapq
from operator import itemgetter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from collections import defaultdict
from nltk.corpus import wordnet, stopwords






TotalDocCount = 1051
courceDocCount = 230
nonCourceDocCount = TotalDocCount - courceDocCount

coursePath = r"C:\Users\sonuk\PycharmProjects\IrAssignment3\course-cotrain-data\fulltext\course"  # input folder path
nonCoursePath = r"C:\Users\sonuk\PycharmProjects\IrAssignment3\course-cotrain-data\fulltext\non-course"
dataSet = r"C:\Users\sonuk\PycharmProjects\IrAssignment3\collection"  # output folder
doc_list = []
labels = []
# Change the directory
os.chdir(coursePath)


# parse html files
def remove_tags(html):
    # parse html content
    soup = BeautifulSoup(html, "html.parser")

    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()

    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)


# Read text File
def read_text_file(file_path, num):
    with open(file_path, 'r') as f:
        s = f.read()
        output = remove_tags(s)
        output_path = dataSet + '\\' + str(num) + '.txt'
        print(output_path)
        write_file = open(output_path, 'w')
        write_file.write(output)
        write_file.close()


# Convert HTML Pages to the text files and store into the Collection Folder

# # iterate through all file in Course Folder
# for file in os.listdir():
#     file_path = f"{coursePath}\{file}"
#     courceDocCount = courceDocCount + 1
#     # call read text file function
#     read_text_file(file_path, courceDocCount)
#
# os.chdir(nonCoursePath)
#
# # iterate through all file in NonCourse Folder
# for file in os.listdir():
#     file_path = f"{nonCoursePath}\{file}"
#     nonCourceDocCount = nonCourceDocCount + 1
#     # call read text file function
#     read_text_file(file_path, nonCourceDocCount)


def makeDocList():
    for i in range(courceDocCount):
        doc_list.append(i + 1)
        labels.append('C')
    for i in range(courceDocCount, TotalDocCount):
        doc_list.append(i + 1)
        labels.append('NC')

    return doc_list


# def spiltIntoTrainTest(doc_list,labels):


def caseFolding(word):
    i = 0
    lower_case = ' '
    while i in range(len(word)):
        # ord function return the ascii value of the character
        ch = ord(word[i])
        if ch > 64 and ch < 91:
            lower_case += chr(ch + 32)
        else:
            lower_case += chr(ch)
        i = i + 1
    return lower_case


def stopword(word):
    file = open(r"C:\Users\sonuk\PycharmProjects\IrAssignment3\Stopword-List.txt","r")
    while 1:
        s_word = file.read()
        # check each word with s_word(stop words) if it is match then return true else false
        if word in s_word:
            return True
        else:
            return False


def doPreprocessing(str1, dict, position):
    # initialize lemmatizer for the corpous
    lemmatizer = WordNetLemmatizer()

    # caseFolding convert string to lower case
    str1 = caseFolding(str1)
    # strip function removes extra spaces from string
    str1 = str1.strip()
    # stopword returns true if string is stop word else false
    check = stopword(str1)
    if not check:
        # performs stemming to the string
        str1 = lemmatizer.lemmatize(str1)

        # if string is already in the dictonary then leave it
        if str1 not in dict:
            dict[str1] = []

        dict[str1].append(position)


def dataPreprocessing(documentID):
    dict = {}
    str1 = ' '
    postion = 1
    brackets = False
    apostrophe_check = False
    j = 0
    file_path = r"C:\Users\sonuk\PycharmProjects\IrAssignment3\collection\\" + str(documentID) + ".txt"
    file1 = open(file_path, "r")
    while 1:
        input = file1.read(1)
        if not input:
            if str1 != " ":
                doPreprocessing(str1, dict, postion)
            break

        # apostrophe_check is true because word like sonu's so due to apostrophe it is true so we have to ignore the s
        if apostrophe_check and input != ' ':
            apostrophe_check = False
            continue

        # # if two words are joined with these hyphen then remove hyphen and combine both to check for one word
        if input == '.' or ord(input) == 92:
            continue

            # checking for the apostrophe
        ch = ord(input)
        if ch == 39:
            apostrophe_check = True
            continue

        # there are some of the invalid characters in this list which should not be in the terms
        if input in ["\\", "/", ":", "*", "?", '"', "<", ">", "|", "%","!"]:
            continue

        # this is for ignoring the abbreviations like information retrieval (IR) so only information and retrieval
        # are there in the dictionary
        if brackets:
            if input == ')':
                brackets = False
            continue
        if input == '(':
            brackets = True
            continue



        elif input == ' ' or input == ',' or input == '-' or ord(input) == 47 or ord(input) == 10 or ord(input) == 9  :  # 10 is the ascii value of the \n or new line

            j = 0
            # 47 is the ascii value of the /
            # if extra or same delimiter come then no need to process only continue
            if str1 == "":
                continue

            doPreprocessing(str1, dict, postion)

            str1 = ""
            postion = postion + 1
        else:
            str1 += input
            j = j + 1
        # when (sonus') this type of situations come
        apostrophe_check = False
    file1.close()
    return dict


def getlistFromDic(dict):
    list = []
    # keys() fucntions contains the keys of the dictionary
    for word in dict.keys():
        list.append(word)
    return list


def constructPostionalIndex(docs):
    postionalIndex = {}
    list = []
    document_id = 0
    for document_id in docs:
        print(document_id)
        # here firstly preprocess the data perfroming tokenization,casefolding, removing stop words and stemming
        temp_dict = dataPreprocessing(document_id)

        # this returns the list of keys from the dictonary
        list = getlistFromDic(temp_dict)

        for i in range(len(list)):
            if list[i] not in postionalIndex:
                postionalIndex[
                    list[i]] = {}  # this assures dictonary for every string to store the documentID and position
            postionalIndex[list[i]][
                document_id] = []  # this assures list for particular string and documentid to store positions

        # this loop runs for how many unique words are there in particular document
        for i in range(len(list)):
            # this loop for appending the positions of particular word in the document
            for j in range(len(temp_dict[list[i]])):
                postionalIndex[list[i]][document_id].append(temp_dict[list[i]][j])
    return postionalIndex


def store_terms(dict):
    for key in dict.keys():
        doc_dict = {}
        if('\x18' not in key):
            for i in dict[key].keys():
                doc_dict[i] = len(dict[key][i])
            # file_path = f'terms/{key}.pkl'
            if("\\" in key):
                key.replace("\\","")
            file_path = r"C:\Users\sonuk\PycharmProjects\IrAssignment3\terms\\" + key + ".pkl"
            file = open(file_path, "wb")
            pickle.dump(doc_dict, file)
            file.close()
        else:
            continue

def docTermTfIdfSocre(str,N):
    doc_dict = {}
    score = {}
    a_file = open(r"C:\Users\sonuk\PycharmProjects\IrAssignment3\terms\\" + str + ".pkl", "rb")
    doc_dict = pickle.load(a_file)

    df = len(doc_dict)

    # calculating idf score taking log base 10
    idf = math.log((N/df),10)

    for i in doc_dict.keys():
        tf = doc_dict[i]
        score[i] = tf * idf

    return score

def storeTfIdfScore(dict):
    tfIdfScores = {}
    score = {}
    for i in dict.keys():
        if i not in tfIdfScores:
            tfIdfScores[i] = {}

    for i in dict.keys():
        if ('\x18' not in i):
            score = docTermTfIdfSocre(i,TotalDocCount)
            tfIdfScores[i] = score

    file = open( r"C:\Users\sonuk\PycharmProjects\IrAssignment3\DocumentTfIdfScores.pkl", "wb")
    pickle.dump(tfIdfScores, file)
    file.close()
    print(tfIdfScores)


def getUpdatedFeaturedDoc(docTfIdfdict, docs):
    termHighestTfIdfScores = {}
    for i in docTfIdfdict.keys():
        if len(docTfIdfdict[i].keys()) > 0:
            doc = max(docTfIdfdict[i][key] for key in docTfIdfdict[i].keys())
            termHighestTfIdfScores[i] = doc

    # print(termHighestTfIdfScores)
    topitems = heapq.nlargest(100, termHighestTfIdfScores.items(), key=itemgetter(1))  # Use .iteritems() on Py2
    topitemsasdict = dict(topitems)
    final_training_Features = []
    for j in docs:
        temp = []
        for i in topitemsasdict.keys():
            if j in docTfIdfdict[i].keys():
                temp.append(1)
            else:
                temp.append(0)
        final_training_Features.append(temp)

    # print(final_training_Features)
    return topitemsasdict,final_training_Features
def getTotalFeatures(frequentNouns):
    final_features_list = frequentNouns
    for i in range(1,TotalDocCount + 1):
        dict = dataPreprocessing(i)
        temp_list = list(dict.keys())
        for j in range(len(temp_list)):
            if(temp_list[j] in frequentNouns):
                if j != (len(temp_list) - 1):
                    final_features_list.append(temp_list[j + 1])
                if j != 0:
                    final_features_list.append(temp_list[j - 1])
    return final_features_list

def findMostFrequentNouns(docTfIdfdict):
    nouns = []
    l = nltk.pos_tag(docTfIdfdict.keys())
    for word,n in l:
        if n == 'NN':
            nouns.append(word)


    nouns_count = {}
    for i in nouns:
        sum = 0
        a_file = open(r"C:\Users\sonuk\PycharmProjects\IrAssignment3\terms\\" + i + ".pkl", "rb")
        doc_dict = pickle.load(a_file)

        for j in doc_dict.keys():
            sum = sum +  doc_dict[j]

        nouns_count[i] = sum

    # print(nouns_count)

    # print(termHighestTfIdfScores)
    topitems = heapq.nlargest(50, nouns_count.items(), key=itemgetter(1))  # Use .iteritems() on Py2
    topitemsasdict = dict(topitems)
    return topitemsasdict




def buildRelation(nouns):
    relation_list = defaultdict(list)

    for k in range(len(nouns)):
        relation = []
        for syn in wordnet.synsets(nouns[k], pos=wordnet.NOUN):
            for l in syn.lemmas():
                relation.append(l.name())
                if l.antonyms():
                    relation.append(l.antonyms()[0].name())
            for l in syn.hyponyms():
                if l.hyponyms():
                    relation.append(l.hyponyms()[0].name().split('.')[0])
            for l in syn.hypernyms():
                if l.hypernyms():
                    relation.append(l.hypernyms()[0].name().split('.')[0])
        relation_list[nouns[k]].append(relation)
    return relation_list



def buildLexicalChain(nouns, relation_list):
    lexical = []
    threshold = 0.5
    for noun in nouns:
        flag = 0
        for j in range(len(lexical)):
            if flag == 0:
                for key in list(lexical[j]):
                    if key == noun and flag == 0:
                        lexical[j][noun] += 1
                        flag = 1
                    elif key in relation_list[noun][0] and flag == 0:
                        syns1 = wordnet.synsets(key, pos=wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos=wordnet.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
                    elif noun in relation_list[key][0] and flag == 0:
                        syns1 = wordnet.synsets(key, pos=wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos=wordnet.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
        if flag == 0:
            dic_nuevo = {}
            dic_nuevo[noun] = 1
            lexical.append(dic_nuevo)
            flag = 1
    return lexical


def eliminateWords(lexical):
    final_chain = []
    while lexical:
        result = lexical.pop()
        if len(result.keys()) == 1:
            for value in result.values():
                if value != 1:
                    final_chain.append(result)
        else:
            final_chain.append(result)
    return final_chain

def naiveBaseBurnolliAlgorithm(X_train, X_test, y_train, y_test):
    # Create a Classifier
    model = BernoulliNB()

    # Train the model using the training sets
    model.fit(X_train, y_train)
    filename = r"C:\Users\sonuk\PycharmProjects\IrAssignment3\bbn.sav"
    pickle.dump(model, open(filename, 'wb'))
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = model.predict(X_test)
    # Model Accuracy
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("F1:", metrics.f1_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))



doc_list = makeDocList()
# print(doc_list)
 # Construct Positional Index then store postings of the terms on harddisk then, calculate their tf idf scores and store them to hard disk
# dict = {}
# dict = constructPostionalIndex(doc_list)
# print(len(dict))
# store_terms(dict)
# storeTfIdfScore(dict)


docTfIdfdict = {}
a_file = open(r"C:\Users\sonuk\PycharmProjects\IrAssignment3\DocumentTfIdfScores.pkl", "rb")
docTfIdfdict = pickle.load(a_file)

print()
print("-------------------Analysis of Tf-Idf Feature-------------------")
print()



#-----------------1st Feature ---------------------
tfIdfFeatures,updatedDocsList = getUpdatedFeaturedDoc(docTfIdfdict,doc_list)
label_encoder = preprocessing.LabelEncoder()
en_label=label_encoder.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(updatedDocsList, en_label, test_size=0.3,random_state=42)
naiveBaseBurnolliAlgorithm(X_train, X_test, y_train, y_test)

print()
print("-------------------Analysis of Topic Terms co-occurrence based Feature-------------------")
print()

#-----------------2nd Feature ---------------------
fiftyFrequentFeatures = findMostFrequentNouns(docTfIdfdict)
totalFeatures = getTotalFeatures(list(fiftyFrequentFeatures.keys()))
final_training_Features = []
for j in doc_list:
    temp = []
    for i in totalFeatures:
        if j in docTfIdfdict[i].keys():
            temp.append(1)
        else:
            temp.append(0)
    final_training_Features.append(temp)

X_train, X_test, y_train, y_test = train_test_split(final_training_Features, en_label, test_size=0.3,random_state=42)
naiveBaseBurnolliAlgorithm(X_train, X_test, y_train, y_test)

print()
print("-------------------Analysis of Lexical Chains Feature-------------------")
print()

#-----------------3rd Feature ---------------------
nouns = []
l = nltk.pos_tag(docTfIdfdict.keys())
for word, n in l:
    if n == 'NN':
        nouns.append(word)

relation = buildRelation(nouns)
lexical = buildLexicalChain(nouns, relation)
final_chain = eliminateWords(lexical)
lexicalChainFeatures = []
for i in final_chain:
    for j in i.keys():
        lexicalChainFeatures.append(j)


# # print(lexicalChainFeatures)
final_training_Features = []
for j in doc_list:
    temp = []
    for i in lexicalChainFeatures:
        if j in docTfIdfdict[i].keys():
            temp.append(1)
        else:
            temp.append(0)
    final_training_Features.append(temp)

X_train, X_test, y_train, y_test = train_test_split(final_training_Features, en_label, test_size=0.2,random_state=42)
naiveBaseBurnolliAlgorithm(X_train, X_test, y_train, y_test)
print()
print("-------------------Analysis of Mixed Feature-------------------")
print()

#-----------------4th Feature ---------------------
totalFeatures =list(set(totalFeatures + list(tfIdfFeatures.keys()) + lexicalChainFeatures))
final_training_Features = []
for j in doc_list:
    temp = []
    for i in totalFeatures:
        if j in docTfIdfdict[i].keys():
            temp.append(1)
        else:
            temp.append(0)
    final_training_Features.append(temp)

X_train, X_test, y_train, y_test = train_test_split(final_training_Features, en_label, test_size=0.2,random_state=42)
naiveBaseBurnolliAlgorithm(X_train, X_test, y_train, y_test)

