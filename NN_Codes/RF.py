import matplotlib.pyplot as plt
import numpy as np
from parse_dataset import Crawler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import coverage_error, label_ranking_average_precision_score, label_ranking_loss

OUTPUTS = 3
TEST_SIZE = 200
FEATURES = 28

N_ESTIMATORS = 11
CRITERION = 'entropy'
MAX_FEATURES = 'auto'
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

# for estimators in range(62,81,2):
#     cov_errs = []
#     lrap_scores = []
#     ranking_losses = []

#     k_fold = KFold(n_splits=10)
#     for train_index, test_index in k_fold.split(X, train_label):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = train_label[train_index], train_label[test_index]

#         test_size = len(X_test)

# train classifier
clf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
    criterion=CRITERION,
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
print "RandomForest - Coverage error: " + str(cover_err)

# label ranking average precision score
# best value is 1
lrap_score = label_ranking_average_precision_score(test_label, scores)
# lrap_scores.append(lrap_score)
print "RandomForest - Label ranking avg precision score: " + str(lrap_score)

# compute label ranking loss
# best value is 0
ranking_loss = label_ranking_loss(test_label, scores)
# ranking_losses.append(ranking_loss)
print "RandomForest - Ranking loss: " + str(ranking_loss)

    # avg_cov_err = np.mean(cover_err)
    # print "RandomForest CV avg coverage error - Estimators " + str(estimators) + " " + str(avg_cov_err)
    # avg_lrap_score = np.mean(lrap_scores)
    # print "RandomForest CV avg lrap score - Estimators " + str(estimators) + " " + str(avg_lrap_score)
    # avg_rank_loss = np.mean(ranking_losses)
    # print "RandomForest CV avg ranking loss - Estimators " + str(estimators) + " " + str(avg_rank_loss)
    # print "\n"