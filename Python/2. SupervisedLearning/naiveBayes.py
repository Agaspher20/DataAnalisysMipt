#%%
from sklearn import datasets
from sklearn.cross_validation import cross_val_score

def write_answer(fileName, answer):
    with open("naiveBayes_" + fileName + ".txt", "w") as fout: #..\..\Results\
        fout.write(str(answer))

digits_data = datasets.load_digits()
digits_X = digits_data.data
digits_y = digits_data.target
print "Digits first row:\n", digits_X[1,:], "\n"

breast_cancer_data = datasets.load_breast_cancer()
breast_cancer_X = breast_cancer_data.data
breast_cancer_y = breast_cancer_data.target
print "Breast cancer first row:\n", breast_cancer_X[1,:], "\n"
#%%
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.cross_validation import cross_val_score

digits_bernoulli = cross_val_score(BernoulliNB(), digits_X, digits_y)
digits_multinomial = cross_val_score(MultinomialNB(), digits_X, digits_y)
digits_gaussian = cross_val_score(GaussianNB(), digits_X, digits_y)

cancer_bernoulli = cross_val_score(BernoulliNB(), breast_cancer_X, breast_cancer_y)
cancer_multinomial = cross_val_score(MultinomialNB(), breast_cancer_X, breast_cancer_y)
cancer_gaussian = cross_val_score(GaussianNB(), breast_cancer_X, breast_cancer_y)
#%%
digits_estimations = [
    ("Bernoulli", digits_bernoulli),
    ("Multinomial", digits_multinomial),
    ("Gaussian", digits_gaussian)
]
cancer_estimations = [
    ("Bernoulli", cancer_bernoulli),
    ("Multinomial", cancer_multinomial),
    ("Gaussian", cancer_gaussian)]
# Question 1
#%%
best_cancer = max(cancer_estimations, key=lambda x: x[1].mean())
write_answer("1", best_cancer[1].mean())
best_cancer
# Question 2
#%%
best_digits = max(digits_estimations, key=lambda x: x[1].mean())
write_answer("2", best_digits[1].mean())
best_digits
# Question 3
#%%
answer_3 = []
if best_cancer[0] == "Bernoulli":
    answer_3.append(1)
    print "Answer 3_1: true"
if best_cancer[0] == "Multinomial":
    answer_3.append(2)
    print "Answer 3_2: true"
if best_digits[0] == "Multinomial":
    answer_3.append(3)
    print "Answer 3_3: true"
if best_cancer[0] == "Gaussian":
    answer_3.append(4)
    print "Answer 3_4: true"
#%%
write_answer("3", " ".join(map(str, answer_3)))
print "Answer 3: ", " ".join(map(str, answer_3))
