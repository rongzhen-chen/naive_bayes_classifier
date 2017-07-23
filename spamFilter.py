import os
from collections import Counter
import numpy as np

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix 

# save the cleaned words, from all emails, into a dictionary
# with counted number as value and word as key
def make_Dictionary(train_dir):
    # the full path of files
    emails = [os.path.join(train_dir,f) for f in sorted(os.listdir(train_dir))]
    all_words = []
    for email in emails:
        with open(email) as e:
            # skip two lines
            next(e)
            next(e)
            for line in e:
                words = line.split()
                all_words += words
    # count the words such as "love:50"; 50 times word "love"
    dictionary=Counter(all_words)

    # clean the dictionary, key is the words, value is the repeated time
    list_to_remove = dictionary.keys()
    for item in list_to_remove:
        if item.isalpha() == False or len(item) == 1:
            del dictionary[item]

    # 3000 most frequently used words in the dictionary
    dictionary = dictionary.most_common(3000)
    
    return dictionary

def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in sorted(os.listdir(mail_dir))]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    #for fil in files:
    for docID, fil in enumerate(files):
        with open(fil) as e:
            next(e)
            next(e)
            for line in e:
                words = line.split()
                for word in words:
                    for wordID,d in enumerate(dictionary):
                        if d[0] == word:
                            features_matrix[docID,wordID] = words.count(word)
    return features_matrix


if __name__ == '__main__':

    path=os.getcwd()  
    trainPath=path+"/train-mails"

    dictionary=make_Dictionary(trainPath)

    train_matrix=extract_features(trainPath)

    train_labels = np.zeros(702)
    # 1 means spam email
    train_labels[351:701] = 1

    model1 = MultinomialNB()
    model2 = LinearSVC()
    model1.fit(train_matrix,train_labels)
    model2.fit(train_matrix,train_labels)

    # Test the unseen mails for Spam
    test_dir = 'test-mails'
    test_matrix = extract_features(test_dir)
    test_labels = np.zeros(260)
    test_labels[130:260] = 1
    result1 = model1.predict(test_matrix)
    result2 = model2.predict(test_matrix)
    print confusion_matrix(test_labels,result1)
    print confusion_matrix(test_labels,result2)
