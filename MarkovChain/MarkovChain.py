import random
import sys
import os

#if comparing two authors
if len(sys.argv) > 4:
    AUTH1 = sys.argv[1] #author 1
    AUTH2 = sys.argv[2] #author 2
    PROB1 = sys.argv[3] #model 1 file
    PROB2 = sys.argv[4] #model 2 file
    RESULT = sys.argv[5] #results file
    Auth2_books = os.listdir(AUTH2)
    two = True #are there two authors provided
else:
    AUTH1 = sys.argv[1] #author 1
    PROB1 = sys.argv[2] #model 1 file
    RESULT = sys.argv[3] #results 1 file
    two = False

#gets list of books in directory
Auth1_books = os.listdir(AUTH1)

stop_words = open("EnglishStopwords.txt", "r")
stop = list(stop_words)
delimiters = ['\n', ',', '.', '?', '!', '"', ';', '"', ':', '(', ')', '&', '-', '[', ']', '_', '”', "' ", " '", '\r', '\t', '*', '&', '^', '-', "'", '—', '“', '’']

#how I store all the data for each model
class model:
    def __init__(self, name, auth, books):
        self.name = name
        self.books = books
        self.author = auth
        self.unigrams = dict()
        self.bigrams = dict()
        self.trigrams = dict()
        self.allwords = list()
        self.binormal = dict() #dictionary for normalized bigrams
        self.trinormal = dict() #dictionary for normalized trigrams
        self.totalwords = 0
        self.sentences = list()
        self.prob = dict()

    def populate(self):
        for book in self.books:
            b = open(self.author + book, 'r')
            for line in b:
                line = line.lower()
                line = removePunc(line)
                words = line.split(' ')
                removeWords(words)
                removeEmpties(words)
                n_grams(self, words, 1) 
            n_grams(self, self.allwords, 2) #bigrams
            n_grams(self, self.allwords, 3) #trigrams
            self.allwords.clear()
        b.close()

#getting all the uni-,bi-,and trigrams
def n_grams(m, words, n):
    if n == 1:
        for word in words:
            if word == '>':
                continue
            value = m.unigrams.get(word)
            if value != None:
                m.unigrams[word] = int(value + 1)
            else:
                m.unigrams[word] = 1
            m.allwords.append(word)
            m.totalwords += 1
    elif n == 2:
        index = 0
        for word in words:
            if index < len(words) - 1:
                if words[index] == '>' or words[index + 1] == '>':
                    continue
                current = words[index] + " " + words[index + 1]

                if m.binormal.get(words[index]) != None:
                    m.binormal[words[index]] += 1
                else:
                    m.binormal[words[index]] = 1

                value = m.bigrams.get(current)
                if value != None:
                    m.bigrams[current] = value + 1
                else:
                    m.bigrams[current] = 1
            index += 1
    
    elif n == 3:
        index = 0
        for word in words:
            if index < len(words) - 2:
                if words[index] == '>' or words[index + 1] == '>' or words[index + 2] == '>':
                    continue
                current = words[index] + " " + words[index + 1] + " " + words[index + 2]

                if m.trinormal.get(words[index] + " " + words[index + 1]) != None:
                    m.trinormal[words[index] + " " + words[index + 1]] += 1
                else:
                    m.trinormal[words[index] + " " + words[index + 1]] = 1

                value = m.trigrams.get(current)
                if value != None:
                    m.trigrams[current] = value + 1
                else:
                    m.trigrams[current] = 1
            index += 1
    
#removing stop words
def removeWords(words):
    for word in stop:
        w = str(word).strip()#[0:word.index("\n")]
        if w in words:
            words.remove(w)
            removeWords(words)    
    
#removing
def removePunc(line):
    for delim in delimiters:
        if len(line) == 0:
            return line
        if delim == '.' or delim == '\n':
            line = line.replace(delim, ' >')
        else:
            line = line.replace(delim, ' ')
    return line
    
#removing any random spaces
def removeEmpties(words):
    if '' in words or ' ' in words:
        for word in words:
            if '' == word:
                words.remove(word)
            if ' ' == word:
                words.remove(word)
        removeEmpties(words)
        
#returns how many of "cat *" bigrams/trigrams
def totalNGrams(m, words, root, grams):
    total = 0
    for w in words:
        if grams == 2:
            total += m.bigrams[root + " " + w]
        else:
            total += m.trigrams[root + " " + w]
    return total
          
#picking the next probable word, don't sort the words first   
def getProbableWord(m, words, root, gram, P):
    current = ""
    
    val = random.uniform(0, 1)
    
    a = ab = totalNGrams(m, words, root, gram)
    if gram == 2:
        for w in words:
            ab = m.bigrams[root + " " + w]
            prob = ab / a
            val -= prob
            if val <= 0:
                P *= prob
                current = w
                break
    else:
        for w in words:
            abc = m.trigrams[root + " " + w]
            prob = abc / ab
            val -= prob
            if val <= 0:
                P *= prob
                current = w
                break
    return current
    
#method for picking the next word in the sentence
def nextWord(m, sentence, root, size, previous, P):
    if size > 0:
        sentence = str(sentence + " " + root)
    else:
        sentence = root
    size += 1
    if size == 20:
        return sentence
    
    next = list()
    if size < 2:
        for w in m.bigrams:
            words = w.split(" ")
            if root == words[0]:
                next.append(words[1])
        newRoot = getProbableWord(m, next, root, 2, P)
    else:
        for w in m.trigrams:
            words = w.split(" ")
            if previous == words[0] and root == words[1]:
                next.append(words[2])
        newRoot = getProbableWord(m, next, previous + " " + root, 3, P)
    return nextWord(m, sentence, newRoot, size, root, P)
    
#creating sentences up to 20 tokens
def createSentence(m):
    sentence = ""
    rand = random.randint(0, len(m.unigrams))
    grams = list(m.unigrams.items())
    word = grams[rand][0]
    P = m.unigrams[word] / m.totalwords
    sentence = nextWord(m, sentence, word, 0, "", P)
    m.sentences.append(sentence)
    m.prob[sentence] = P
    
#gets probability of the sentences in each model or same model if m = o
def compare(m, o, r):
    for s in m.sentences:
        r.write(s + '\n')
        r.write(m.name + ' Model: ' + str(m.prob[s]) +'\n')
        print(s)
        print(m.name + ' Model: ' + str(m.prob[s]))
        if len(sys.argv) > 4:
            P = 1
            index = 0
            words = s.split()
            for w in words:
                if index == 0:
                    if o.unigrams.get(w):
                        P *= o.unigrams[w] / o.totalwords
                    else:
                        P *= .00001
                elif index == 1:
                    if o.bigrams.get(words[0] + " " + w):
                        P *= o.bigrams[words[0] + " " + w] / o.binormal(words[0])
                    else:
                        P *= .00001
                else:
                    if o.trigrams.get(words[index - 2] + " " + words[index - 1] + " " + w):
                        P *= o.trigrams[words[index - 2] + " " + words[index - 1] + " " + w] / o.trinormal[words[index - 2] + " " + words[index - 1]]  
                    else:
                        P *= .00001
                index += 1
            r.write(o.name + ' Model: ' + str(P) + '\n\n')
            print(o.name + ' Model: ' + str(P))            
    
#outputs to the PROB files
def printProb(m, pfile):
    file = open(pfile, 'w')
    for u in m.unigrams:
        value = round(m.unigrams[u] / m.totalwords, 10)
        file.write(u + ": " + str(value) + '\n')
    for b in m.bigrams:
        words = b.split()
        value = round(m.bigrams[b] / m.binormal[words[0]], 10)
        file.write(b + ": " + str(value) + '\n')
    for t in m.trigrams:
        words = t.split()
        value = round(m.trigrams[t] / m.trinormal[words[0] + " " + words[1]], 10)
        file.write(t + ": " + str(value) + '\n')
    file.close()

m1 = model(AUTH1[0:len(AUTH1)-1], AUTH1, Auth1_books)
m1.populate()
printProb(m1, PROB1)
for n in range(10):
    createSentence(m1)

if two:
    m2 = model(AUTH2[0:len(AUTH2)-1], AUTH2, Auth2_books)
    m2.populate()
    printProb(m2, PROB2)
    for n in range(10):
        createSentence(m2)    
    results = open(RESULT, 'w')
    print(m1.name + " sentences\n")
    results.write(m1.name + " sentences\n")
    compare(m1, m2, results)
    print('\n' + m2.name + " sentences\n")
    results.write('\n' + m2.name + " sentences\n")
    compare(m2, m1, results)
    results.close()
else:
    results = open(RESULT, 'w')
    print(m1.name + " sentences\n")
    compare(m1, m2)
    results.close()

    