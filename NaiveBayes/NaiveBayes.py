import sys
import csv

arg1 = sys.argv[0] #file
arg2 = sys.argv[1] #train file
arg3 = sys.argv[2] #test file
arg4 = sys.argv[3] #model file
arg5 = sys.argv[4] #results file

TRAIN = arg2
TEST = arg3
MODEL = arg4
RESULTS = arg5

train_data = dict()
train_file = open(TRAIN, "r")
train_values = csv.reader(train_file, delimiter = ',')

test_data = dict()
test_file = open(TEST, "r")
test_values = csv.reader(test_file, delimiter = ',')

i = 0
last_index = 0
last_column = ""
names = None
for row in train_values:
    if i == 0:
        #gets the names/first row of the csv file and the last index
        names = row
        last_index = len(names) - 1
        last_column = names[-1]
    else:
        #checks the rest of the rows to find the frequency of each characteristic
        other = row
        j = 0
        for v in other:
            name = names[j]
            if j < last_index:
                key = name + '-'+ v + '-' +other[-1]

                pair = train_data.get(key)
                if pair:
                    train_data[key] = int(pair + 1)
                else:
                    train_data[key] = 1
            else:
                key = name + '-' + v
                pair = train_data.get(key)
                if pair:
                    val = pair + 1
                    train_data[key] = int(val)
                else:
                    train_data[key] = 1
            j += 1
    i += 1

#setting the frequency of class 0 and 1  
class1 = train_data[last_column + '-1']
class0 = train_data[last_column + '-0']

probabilities = dict()
for var in train_data:
    if last_column in var:
        #if in the class column just different to get the probability of each
        data = train_data[var]
        total = class1 + class0
        probabilities[var] = round(data/total, 4)
    else:
        #else P(characteristic|class) = character^class/class
        data = train_data[var]
        total = train_data[last_column + '-' + var[-1]]
        probabilities[var] = round(data/total, 4)


i = 0
predicted_yes = 0
predicted_no = 0
actual_yes = 0
actual_no = 0
correct = 0
wrong = 0
false_positive = 0 #predicted yes, but actually no
true_positive = 0 #predicted yes, and actually yes
false_negative = 0 #predicted no, but actually yes
true_negative = 0 #predicted no, and actually no
results = open(RESULTS, 'w')
#for each row in the test file, multiply everything to get the probability of class 0 and 1
for row in test_values:
    predict_yes = 1
    predict_no = 1
    if i > 0:
        j = 0
        for v in row:
            if j == last_index:
                nprob = probabilities[last_column + '-0']
                yprob = probabilities[last_column + '-1']
                predict_no *= nprob
                predict_yes *= yprob
                final_no = predict_no / (predict_no + predict_yes)
                final_yes = predict_yes / (predict_no + predict_yes)
                if final_yes > final_no:
                    results.write("Line " + str(i) + " prediction: " + 'class 1, ')
                    predicted_yes += 1
                    if row[-1] == '1':
                        results.write("actual: class 1\n")
                        actual_yes += 1
                        true_positive += 1
                        correct += 1
                    else:
                        results.write("actual: class 0\n")
                        actual_no += 1
                        false_positive += 1
                        wrong += 1
                else:
                    results.write("Line " + str(i) + " prediction: " + 'class 0, ')
                    predicted_no += 1
                    if row[-1] == '0':
                        results.write("actual: class 0\n")
                        true_negative += 1
                        actual_no += 1
                        correct += 1
                    else:
                        results.write("actual: class 1\n")
                        actual_yes += 1
                        false_negative += 1
                        wrong += 1
            else:
                key0 = str(names[j] + '-' + v + '-0')
                key1 = str(names[j] + '-' + v + '-1')
                if probabilities.get(key0):
                    predict_no *= probabilities[key0]
                if probabilities.get(key1):
                    predict_yes *= probabilities[key1]
            j += 1
    i += 1
    
mod = open(MODEL, "w")
probs = list(probabilities.keys())
probs.sort()
for p in probs:
    mod.write('P('+ str(p).strip() +') = ' + str(probabilities[p]) + "\n")
mod.close()

results.write("\nYes " + str(actual_yes) + " : No " + str(actual_no))
results.write("\nAN = Actual No,\nAY = Actual Yes,\nPN = Predicted No,\nPY = Predicted Yes\n")
results.write("   PN PY\n")
results.write("AN " + str(true_negative) + " " + str(false_positive) +'\n')
results.write("AY " + str(false_negative) + " " + str(true_positive))
results.close()
    
print("Yes " + str(actual_yes) + " : No " + str(actual_no))
print("AN = Actual No,\nAY = Actual Yes,\nPN = Predicted No,\nPY = Predicted Yes\n")
print("   PN PY")
print("AN " + str(true_negative) + " " + str(false_positive))
print("AY " + str(false_negative) + " " + str(true_positive))