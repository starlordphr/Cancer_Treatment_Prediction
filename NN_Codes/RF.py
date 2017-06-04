import numpy as np
from parse_dataset import Crawler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import coverage_error, label_ranking_average_precision_score, label_ranking_loss

OUTPUTS = 3
TEST_SIZE = 200
FEATURES = 28

N_ESTIMATORS = 5
CRITERION = 'entropy'
MAX_FEATURES = 'auto'
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 50


objCrawler = Crawler("FinalCancer_Data.csv")
data = objCrawler.parse_input()

test_data = data[:TEST_SIZE,:FEATURES]
test_label = data[:TEST_SIZE, -OUTPUTS:]

train_data = data[TEST_SIZE:, :FEATURES]
train_label = data[TEST_SIZE:, -OUTPUTS:]

# normalize dataset
norm = Normalizer()
X = norm.fit_transform(train_data)

# train classifier
clf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
    criterion=CRITERION,
    max_features=MAX_FEATURES,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT).fit(X, train_label)

# make predictions on test data
Y = norm.transform(test_data)
predicted = clf.predict_proba(Y)
scores = np.reshape(np.concatenate((predicted[0][:,1], predicted[1][:,1], predicted[2][:,1])), (TEST_SIZE, OUTPUTS))

# compute error for multilabel ranking
# coverage error
# the best value is equal to the avg number of labels in test_label per sample
# i.e. 2.0550000000000002
cover_err = coverage_error(test_label, scores)
print "Coverage error: " + str(cover_err)

# label ranking average precision score
# best value is 1
lrap_score = label_ranking_average_precision_score(test_label, scores)
print "Label ranking avg precision score: " + str(lrap_score)

# compute label ranking loss
# best value is 0
ranking_loss = label_ranking_loss(test_label, scores)
print "Ranking loss: " + str(ranking_loss)
