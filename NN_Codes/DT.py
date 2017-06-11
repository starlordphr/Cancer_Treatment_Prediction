import matplotlib.pyplot as plt
import numpy as np
from parse_dataset import Crawler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import coverage_error, label_ranking_average_precision_score, label_ranking_loss

OUTPUTS = 3
TEST_SIZE = 200
FEATURES = 28

CRITERION = 'entropy'
MAX_FEATURES = None
MAX_DEPTH = 6
MIN_SAMPLES_SPLIT = 100

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

# 10-Fold CV
# for criterion in ['gini', 'entropy']:
#     for max_depth in range(3,15):
#         cov_errs = []
#         lrap_scores = []
#         ranking_losses = []

#         k_fold = KFold(n_splits=10)
#         for train_index, test_index in k_fold.split(X, train_label):
#             X_train, X_test = X[train_index], X[test_index]
#             y_train, y_test = train_label[train_index], train_label[test_index]

#             test_size = len(X_test)

# train classifier
clf = DecisionTreeClassifier(criterion=CRITERION,
    max_features=MAX_FEATURES,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT).fit(X, train_label)

# make predictions on test data
predicted = clf.predict_proba(Y)
scores = np.reshape(np.concatenate((predicted[0][:,1], predicted[1][:,1], predicted[2][:,1])), (TEST_SIZE, OUTPUTS))

# compute error for multilabel ranking
# coverage error
# the best value is equal to the avg number of labels in test_label per sample
# i.e. 2.0550000000000002
cover_err = coverage_error(test_label, scores)
# cov_errs.append(cover_err)
print "DecisionTree - Coverage error: " + str(cover_err)

# label ranking average precision score
# best value is 1
lrap_score = label_ranking_average_precision_score(test_label, scores)
# lrap_scores.append(lrap_score)
print "DecisionTree - Label ranking avg precision score: " + str(lrap_score)

# compute label ranking loss
# best value is 0
ranking_loss = label_ranking_loss(test_label, scores)
# ranking_losses.append(ranking_loss)
print "DecisionTree - Ranking loss: " + str(ranking_loss)

        # avg_cov_err = np.mean(cover_err)
        # print "DecisionTree CV avg coverage error - " + criterion + " " + str(max_depth) + " " + str(avg_cov_err)
        # avg_lrap_score = np.mean(lrap_scores)
        # print "DecisionTree CV avg lrap score - " + criterion + " " + str(max_depth) + " " + str(avg_lrap_score)
        # avg_rank_loss = np.mean(ranking_losses)
        # print "DecisionTree CV avg ranking loss - " + criterion + " " + str(max_depth) + " " + str(avg_rank_loss)
