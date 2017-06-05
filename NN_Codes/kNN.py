import matplotlib.pyplot as plt
import numpy as np
from parse_dataset import Crawler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import coverage_error, label_ranking_average_precision_score, label_ranking_loss

NEIGHBORS = 202
OUTPUTS = 3
TEST_SIZE = 200
FEATURES = 28

objCrawler = Crawler("FinalCancer_Data.csv")
data = objCrawler.parse_input()

test_data = data[:TEST_SIZE,:FEATURES]
test_label = data[:TEST_SIZE, -OUTPUTS:]

train_data = data[TEST_SIZE:, :FEATURES]
train_label = data[TEST_SIZE:, -OUTPUTS:]

# normalize dataset
norm = Normalizer()
X = norm.fit_transform(train_data)
Y = norm.transform(test_data)

# lrap_scores = []
# ranking_losses = []
# for i in range(175,251):

# train classifier
clf = KNeighborsClassifier(n_neighbors=NEIGHBORS, weights='distance').fit(X, train_label)
# make predictions on test data
predicted = clf.predict_proba(Y)
scores = np.reshape(np.concatenate((predicted[0][:,1], predicted[1][:,1], predicted[2][:,1])), (TEST_SIZE, OUTPUTS))

# compute error for multilabel ranking
# coverage error
# the best value is equal to the avg number of labels in test_label per sample
# i.e. 2.0550000000000002
cover_err = coverage_error(test_label, scores)
print "kNN - Coverage error: " + str(cover_err)

# label ranking average precision score
# best value is 1
lrap_score = label_ranking_average_precision_score(test_label, scores)
# lrap_scores.append(lrap_scores)
print "kNN - Label ranking avg precision score: " + str(lrap_score)

# compute label ranking loss
# best value is 0
ranking_loss = label_ranking_loss(test_label, scores)
# ranking_losses.append(ranking_loss)
print "kNN - Ranking loss: " + str(ranking_loss)

# plt.plot(range(175,251), ranking_losses)
# plt.show()
