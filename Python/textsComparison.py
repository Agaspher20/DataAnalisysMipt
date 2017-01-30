import re
import numpy as np
import scipy.spatial

sentencesPath = 'sentences.txt'
resultPath = 'submission-1.txt'

def appendToken(dictionary, token):
    tokensCount = 1
    if(token in dictionary):
        tokensCount = dictionary[token] + 1
    dictionary[token] = tokensCount

def tokenizeLine(line):
    return [token for token in re.split('[^a-z]', line.lower()) if token.strip()]

allTokens = {}
tokensByLines = []
lines = []

sentencesFile = open(sentencesPath, 'r')
for line in sentencesFile:
    valueableTokens = []
    lineTokens = {}
    for token in tokenizeLine(line):
        appendToken(allTokens, token)
        appendToken(lineTokens, token)
        valueableTokens.append(token)
    tokensByLines.append(lineTokens)
    lines.append(line)
sentencesFile.close()

tokens = []
for lineTokens in tokensByLines:
    tokensVector = []
    for token in allTokens:
        if token in lineTokens:
            tokensVector.append(lineTokens[token])
        else:
            tokensVector.append(0)
    tokens.append(tokensVector)

tokensMatrix = np.array(tokens)

firstSentence = tokensMatrix[0]
otherSentences = tokensMatrix[1:]

index = 1

sentencesWithDistance = []
for sentence in otherSentences:
    distance = scipy.spatial.distance.cosine(firstSentence, sentence)
    sentencesWithDistance.append((index, lines[index], distance))
    index = index + 1

print 'Matrix shape: ', tokensMatrix.shape[0], ' ', tokensMatrix.shape[1]
print 'First sentence: ', lines[0]
sortedSentences = sorted(sentencesWithDistance, key = lambda tuple: tuple[2])
for sentenceData in sortedSentences[0:2]:
    index,sentence,distance = sentenceData
    print 'Distance: ', distance, ' Index: ', index
    print sentence, '\n'

result = ' '.join([str(sentenceData[0]) for sentenceData in sortedSentences[0:2]])
print 'Result: ', result

submissionFile = open(resultPath, 'w')
submissionFile.write(result)
submissionFile.close()
