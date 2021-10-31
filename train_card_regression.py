import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as skmetrics
from utils import get_point_edges

num_cards = 52
path_CSV = "data/cardpoints.csv"
card_data = pd.read_csv(path_CSV)
points_visible = [k for k in list(card_data.columns.values) if '_visible' in k]
points_coordinates = [k for k in list(card_data.columns.values) if ('_x' in k) or ('_y' in k)]

# Adding new features: computing the edges (i.e. Euclidean distance between keypoints)
card_data, edge_cols = get_point_edges(card_data)

# Choosing the features
feature_cols = points_coordinates + points_visible + edge_cols

# Validation set
val_pct = 0.25
val_data = None
for i in range(num_cards):
    df = card_data[card_data['cards_counted'] == i+1]
    if len(df) > 0:
        val_size = int(len(df) * val_pct)
        sample_df = df.sample(n=val_size)
        if not val_data is None:
            val_data = pd.concat([val_data, sample_df])
        else:
            val_data = sample_df
val_images = list(val_data['imageName'].unique())
# I was lazy and stopped annotating at 4 cards, so we don't really have 52 classes (one per card)
lowest_count = val_data['cards_counted'].min()
X_val = val_data[feature_cols]
y_val = val_data['cards_counted'] - lowest_count

# Training set
train_data = card_data[card_data['imageName'].isin(val_images) == False]
X_train = train_data[feature_cols]
y_train = train_data['cards_counted'] - lowest_count

# Fitting a MaxEnt model to predict the probability of the number of cards
# This was very useful https://cla2019.github.io/scikit_classification.pdf
clf = LogisticRegression(random_state=0, max_iter=300).fit(X_train, y_train)
y_prob = clf.predict_proba(X_val)
y_cards = np.matmul(y_prob, np.array([range(lowest_count, num_cards+1)]).T).astype(int).squeeze()

# Metrics: Accuracy and Mean Absolute Error (MAE)
acc = skmetrics.accuracy_score(y_val+lowest_count, y_cards)
mae = skmetrics.mean_absolute_error(y_val+lowest_count, y_cards)
print("Acc: {} | MAE: {}".format(round(acc, 4), int(mae)))

