# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 20:17:39 2024

@author: zhaoez
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 8 stroke types/labels, combined L and R
label_mapping = {
    "00 breastroke": "breastroke",
    "01 butterfly": "butterfly",
    "02 backstroke L": "backstroke",
    "03 backstroke R": "backstroke",
    "04 freestyle L": "freestyle",
    "05 freestyle R": "freestyle",
    "06 flipturn": "flipturn",
    "07 open turn": "open turn",
    "08 pushoff": "pushoff",
    "09 startDive( .from a block)": "startDive(from a block)"
}

main_folder = r"C:\Users\zhaoez\Desktop\stroke classification business report\strokeanalysis - translated"
# stores the sensor data
data_list = []
# stores the coresponding label
label_list = []
# limit max timesteps as # of data in each file was inconsistent, standardizes the data
max_timesteps = 500

def preprocess(filepath):
    df = pd.read_csv(filepath)
    # removes the time entry as that will not be used to directly train models
    df.drop(columns=["hh:mm:ss.ms"], inplace=True, errors='ignore')
    # returns IMU sensor data as an array
    return df.values

# goes through all folders and files
for root, dirs, files in os.walk(main_folder):
    for dir in dirs:
        mapped_label = label_mapping.get(dir, dir)
        folder_path = os.path.join(root, dir)
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.endswith(".csv"):
                # adds file data to the list after removing the time
                try:
                    sensor_data = preprocess(file_path)
                    data_list.append(sensor_data)
                    label_list.append(mapped_label)
                except Exception as e:
                    print(f"error processing {file_path}: {e}")

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

data_array = pad_sequences(data_list, maxlen=max_timesteps, dtype='float32', padding='post', truncating='post')
# encodes and classifies the labels as integers
labels = pd.factorize(np.array(label_list))[0]
# one-hot encoding for classification
labels_one_hot = to_categorical(labels)
# number of data files for each stroketype/label
label_counts = pd.Series(label_list).value_counts()
print(label_counts)

# bar plot for count of data for each stroke type
plt.figure()
sns.barplot(x=label_counts.index, y=label_counts.values)
plt.title("distribution of data by stroke type")
plt.xlabel("swim classification")
plt.ylabel("count")
plt.xticks(rotation=45)
plt.show()

# flattens into a 2D array where it goes from (data(286), timesteps(500), features(9)) to (data(286), timesteps*features)
data_df = pd.DataFrame(data_array.reshape(data_array.shape[0], -1))

# list of sensor features that will be the column
feature_columns = ['gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ', 'magX', 'magY', 'magZ']
feature_names = [f"{sensor}_{i+1}" for sensor in feature_columns for i in range(max_timesteps)]
data_df.columns = feature_names
data_df['label'] = label_list

# plot style
plt.style.use('seaborn-v0_8')
n_sensors = len(feature_columns)
n_cols = 3
n_rows = (n_sensors + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
fig.suptitle("distribution of sensor data across stroke types", fontsize=16, y=1.02)
axes = axes.flatten()
colors = sns.color_palette("husl", n_colors=len(pd.Series(label_list).unique()))

# bar plot for each sensor data
for idx, sensor in enumerate(feature_columns):
    sensor_idx = feature_columns.index(sensor)
    
    # plot for each stroke type
    for stroke_idx, stroke_type in enumerate(pd.Series(label_list).unique()):
        stroke_indices = [i for i, label in enumerate(label_list) if label == stroke_type]
        sensor_values = data_array[stroke_indices, :, sensor_idx].flatten()
        axes[idx].hist(sensor_values, 
                      bins=50,
                      alpha=0.5,
                      label=stroke_type,
                      color=colors[stroke_idx])
    
    axes[idx].set_title(f"{sensor} distribution")
    axes[idx].set_xlabel(f"{sensor} value")
    axes[idx].set_ylabel("count")
    
    if idx == 0:
        axes[idx].legend(title="stroke type", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[idx].get_legend().remove() if axes[idx].get_legend() else None

    # log y axis for smaller counts to be visible
    axes[idx].set_yscale('log')
    axes[idx].tick_params(axis='x', rotation=45)
    
plt.tight_layout()
plt.show()

# calculates the mean, std, min/max, range, percentage of data that is 0 for each sensor data
feature_stats = []
for idx, sensor in enumerate(feature_columns):
    sensor_values = data_array[:, :, idx].flatten()
    stats = {
        'sensor': sensor,
        'mean': np.mean(sensor_values),
        'std': np.std(sensor_values),
        'min': np.min(sensor_values),
        'max': np.max(sensor_values),
        'range': np.max(sensor_values) - np.min(sensor_values),
        'zero_percentage': np.mean(sensor_values == 0) * 100
    }
    feature_stats.append(stats)

stats_df = pd.DataFrame(feature_stats)
print("sensor statistics:")
print(stats_df)

# bar plot for each sensor feature
plt.figure(figsize=(20, 15))
n_sensors = len(feature_columns)
n_cols = 3
n_rows = (n_sensors + n_cols - 1) // n_cols

for idx, sensor in enumerate(feature_columns):
    plt.subplot(n_rows, n_cols, idx + 1)
    sensor_values = data_array[:, :, idx].flatten()
    counts, bins, _ = plt.hist(sensor_values, bins=50, alpha=0)
    plt.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.7)
    
    plt.title(f"{sensor} distribution")
    plt.xlabel("value")
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

plt.tight_layout()
plt.show()

# correlation matrix of the sensor values
reshaped_data = data_array.reshape(-1, 9)
correlation_matrix = np.corrcoef(reshaped_data.T)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            xticklabels=feature_columns,
            yticklabels=feature_columns,
            annot=True, 
            cmap='coolwarm',
            vmin=-1, 
            vmax=1,
            center=0)
plt.title("correlation matrix of sensor features")
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

# 80:20 data split, reproducibility
X_train, X_test, y_train, y_test = train_test_split(data_array, labels_one_hot, test_size=0.2, random_state=42)
# check shapes of data arrays (sanity)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 3-D -> 2D for PCA
X_flat_train = X_train.reshape(X_train.shape[0], -1)
# standardize features
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_flat_train)
# PCA
pca = PCA()
X_pca_train = pca.fit_transform(X_scaled_train)
# explained variance ratio = eigenvalue / sum of all eigenvalues
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# scree
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
plt.title('scree')
plt.xlabel('principal component')
plt.ylabel('explained variance ratio')
plt.grid(True)

# cumulative explained variance 
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'ro-')
plt.axhline(y=0.90, color='k', linestyle='--', label='90% threshold')
plt.title('plot of cumulative explained variance ratio')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance ratio')
plt.legend()
plt.tight_layout()
plt.show()

X_flat_test = X_test.reshape(X_test.shape[0], -1)
X_scaled_test = scaler.transform(X_flat_test)
X_pca_test = pca.transform(X_scaled_test)

n_components_90 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1
print(f"\ncomponents needed for 90% variance: {n_components_90}")
# PCA for the number of components needed for 90%
# 26 PC <=> sum of first 26 eigenvalues is 90%
pca_final = PCA(n_components=n_components_90)
X_train_pca = pca_final.fit_transform(X_scaled_train)
X_test_pca = pca_final.transform(X_scaled_test)
print(f"original data shape: {X_train.shape}")
print(f"PCA transformed data shape: {X_train_pca.shape}")

# loadings vs eigen values: eigen are the variance explained by PC while loadings are contributions
# contribution of original variables to principal component
loadings = pca_final.components_
# 9 sensors from IMU
n_features = X_train.shape[2]
# 500 timesteps
n_timesteps = X_train.shape[1]
sensor_loadings = loadings.reshape(n_components_90, n_timesteps, n_features)
avg_sensor_loadings = np.mean(sensor_loadings, axis=1)

# plot for loadings
plt.figure(figsize=(12, 8))
sns.heatmap(avg_sensor_loadings, 
            xticklabels=['gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ', 'magX', 'magY', 'magZ'],
            yticklabels=[f'PC{i+1}' for i in range(n_components_90)],
            cmap='coolwarm', center=0, annot=True)
plt.title('average feature contributions to principal components')
plt.tight_layout()
plt.show()

y_train_labels = np.argmax(y_train, axis=1)
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', 
                                   classes=np.unique(y_train_labels), 
                                   y=y_train_labels)
class_weight_dict = dict(zip(np.unique(y_train_labels), class_weights))
print("\nclass weights:")
for class_label, weight in class_weight_dict.items():
    print(f"class {class_label}: {weight:.6f}")
    
plt.figure(figsize=(10, 5))
class_labels = list(class_weight_dict.keys())
weights = list(class_weight_dict.values())

plt.bar(class_labels, weights)
plt.title("class weights distribution")
plt.xlabel("class")
plt.ylabel("weight")

for i, weight in enumerate(weights):
    plt.text(i, weight, f'{weight:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# cross validation for non-temporal models
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_test_labels = np.argmax(y_test, axis=1)
X_train_balanced = X_train_pca
y_train_balanced = y_train_labels

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# logistic regression
# multi-class classification
lr = LogisticRegression(max_iter=1000, C=0.001, 
                       class_weight=class_weight_dict)
lr.fit(X_train_balanced, y_train_balanced)
lr_train_pred = lr.predict(X_train_balanced)
lr_test_pred = lr.predict(X_test_pca)
print("\nlr training:")
print(classification_report(y_train_balanced, lr_train_pred, digits=6))

from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier

param_dist = {
    # ways to icnrease regularization to avoid overfitting
    'n_estimators': randint(200, 400),  # increase trees 
    'max_depth': randint(3, 6),  # reduce max depth
    'min_samples_split': randint(30, 50),  # increase split threshold
    'min_samples_leaf': randint(10, 25),  # increase leaf size
    'max_features': ['sqrt', 'log2'],
    'max_leaf_nodes': randint(15, 30),  # reduce max leaves
    'min_impurity_decrease': uniform(0.0001, 0.01),  # require meaningful splits
    'bootstrap': [True]  # keep bootstrap for better generalization
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, 
                          class_weight=class_weight_dict,
                          oob_score=True),  # enable out-of-bag score
    param_distributions=param_dist,
    n_iter=100,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

rf_random.fit(X_train_balanced, y_train_balanced)
print("\nbest param:")
print(rf_random.best_params_)
print("\ncv score:", rf_random.best_score_)

best_rf = rf_random.best_estimator_
rf_train_pred = best_rf.predict(X_train_balanced)
rf_test_pred = best_rf.predict(X_test_pca)

print("\nrf training:")
print(classification_report(y_train_balanced, rf_train_pred, digits=6))

from xgboost import XGBClassifier
# xgb
#old xgb without early stopping
'''
xgb = XGBClassifier(random_state=42, max_depth=4,
                    min_child_weight=7,
                    reg_alpha=0.5,
                    reg_lambda=2)
xgb.fit(X_train_balanced, y_train_balanced)
xgb_train_pred = xgb.predict(X_train_balanced)
xgb_test_pred = xgb.predict(X_test_pca)
print("\nxgb training:")
print(classification_report(y_train_balanced, xgb_train_pred, digits=6))
'''
# need to split training data for validation for early stopping
X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
    X_train_balanced, y_train_balanced, 
    test_size=0.2, 
    random_state=42
)

xgb = XGBClassifier(
    random_state=42, 
    max_depth=4, # shallow trees 
    min_child_weight=7,
    reg_alpha=0.5, # l1
    reg_lambda=2, #l2
    # stop if no improvement for 10 rounds
    early_stopping_rounds=10 
)

# fit with validaiton data
xgb.fit(
    X_train_xgb, 
    y_train_xgb,
    eval_set=[(X_val_xgb, y_val_xgb)],
    verbose=False
)

xgb_train_pred = xgb.predict(X_train_balanced)
xgb_test_pred = xgb.predict(X_test_pca)
print("\nxgb training:")
print(classification_report(y_train_balanced, xgb_train_pred, digits=6))

from keras.layers import Input
from keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow import keras
from keras_tuner import RandomSearch

# optimizer for temporal model
optimizer = keras.optimizers.Adam(
    learning_rate=1e-3,  # default learning rate
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False
)

# temporal model
# hyperparameter tuning
def build_model(hp):
    '''
    Hybrid CNN and LSTM DL model for time series classification
    1-2 CNN layers (2nd optional)
    2 LSTM layers
    1-2 Dense layers (2nd optional)
    Final layer
    '''
    # input layer
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = inputs
    
    # CNN layers
    x = Conv1D(
        filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Int('conv1_kernel', min_value=2, max_value=5),
        activation='relu'
    )(x)
    x = MaxPooling1D(
        pool_size=hp.Int('pool1_size', min_value=2, max_value=4)
    )(x)
    x = BatchNormalization()(x)
    
    # 2nd CNN layer
    if hp.Boolean('add_conv_layer'):
        x = Conv1D(
            filters=hp.Int('conv2_filters', min_value=16, max_value=64, step=16),
            kernel_size=hp.Int('conv2_kernel', min_value=2, max_value=5),
            activation='relu'
        )(x)
        x = MaxPooling1D(
            pool_size=hp.Int('pool2_size', min_value=2, max_value=4)
        )(x)
        x = BatchNormalization()(x)
    
    # LSTM layers
    x = LSTM(
        units=hp.Int('lstm1_units', min_value=32, max_value=128, step=32),
        return_sequences=True
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1))(x)
    
    x = LSTM(
        units=hp.Int('lstm2_units', min_value=16, max_value=64, step=16)
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1))(x)
    
    # dense layers
    x = Dense(
        units=hp.Int('dense1_units', min_value=16, max_value=64, step=16),
        activation='relu'
    )(x)
    
    if hp.Boolean('add_dense_layer'):
        x = Dense(
            units=hp.Int('dense2_units', min_value=8, max_value=32, step=8),
            activation='relu'
        )(x)
    
    # output layer
    outputs = Dense(8, activation='softmax')(x)

    optimizer = keras.optimizers.Adam(
        learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    )
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

X_temporal = data_array.reshape(data_array.shape[0], -1, 9)
X_train_temporal, X_test_temporal, y_train_temporal, y_test_temporal = train_test_split(
    X_temporal, labels_one_hot, test_size=0.2, random_state=42
)
y_train_temporal_labels = np.argmax(y_train_temporal, axis=1)
temporal_class_weights = compute_class_weight('balanced', 
                                            classes=np.unique(y_train_temporal_labels), 
                                            y=y_train_temporal_labels)
temporal_class_weight_dict = dict(zip(np.unique(y_train_temporal_labels), temporal_class_weights))

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    # change to lower value 
    max_trials=20,
    directory='keras_tuner',
    project_name='cnn_lstm_tuning'
)

# early stop
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# hyperparameter search
tuner.search(
    X_train_temporal,
    y_train_temporal,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    class_weight=temporal_class_weight_dict
)

best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
history = best_model.fit(
    X_train_temporal,
    y_train_temporal,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    class_weight=temporal_class_weight_dict,
    verbose=0
)

import tensorflow as tf

print("\nbest hyperparameters:")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")
print("\nnon-temporal models:")
print("\nlogistic regression:")
print(classification_report(y_test_labels, lr.predict(X_test_pca), digits=6, zero_division=0))
print("\nrandom forest:")
print(classification_report(y_test_labels, rf_random.predict(X_test_pca), digits=6, zero_division=0))
print("\nXGBoost:")
print(classification_report(y_test_labels, xgb.predict(X_test_pca), digits=6, zero_division=0))
print("\nCNN-LMST hybrid w/hyperparameter tuning:")
@tf.function(reduce_retracing=True)
def predict_fn(model, data):
    return model(data, training=False)

def make_predictions(model, data):
    predictions = []
    batch_size = 32
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        pred = predict_fn(model, batch)
        predictions.append(pred)
    
    return tf.concat(predictions, axis=0)

test_pred = make_predictions(best_model, X_test_temporal)
test_pred = test_pred.numpy()
print(classification_report(y_test_temporal.argmax(axis=1), test_pred.argmax(axis=1), digits=6, zero_division=0))

# statistical summary of feature importance
rf_importance_pairs = []
feature_names = ['gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ', 'magX', 'magY', 'magZ']
for name, importance in zip(feature_names, best_rf.feature_importances_):
    rf_importance_pairs.append((name, importance))

sorted_importance = sorted(rf_importance_pairs, key=lambda x: x[1], reverse=True)

print("\nrandom forest feature importance:")
for sensor, importance in sorted_importance:
    print(f"{sensor}: {importance:.6f}")
    
# confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    stroke_types = list(dict.fromkeys(label_mapping.values()))
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] > len(stroke_types):
        backstroke_indices = [i for i, label in enumerate(label_mapping.values()) 
                            if label == 'backstroke']
        freestyle_indices = [i for i, label in enumerate(label_mapping.values()) 
                           if label == 'freestyle']
        
        if len(backstroke_indices) > 1:
            cm[backstroke_indices[0]] += cm[backstroke_indices[1]]
            cm = np.delete(cm, backstroke_indices[1], axis=0)
            cm[:, backstroke_indices[0]] += cm[:, backstroke_indices[1]]
            cm = np.delete(cm, backstroke_indices[1], axis=1)
        
        if len(freestyle_indices) > 1:
            cm[freestyle_indices[0]] += cm[freestyle_indices[1]]
            cm = np.delete(cm, freestyle_indices[1], axis=0)
            cm[:, freestyle_indices[0]] += cm[:, freestyle_indices[1]]
            cm = np.delete(cm, freestyle_indices[1], axis=1)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=stroke_types,
                yticklabels=stroke_types)
    plt.title(f'confusion matrix - {model_name}')
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_test_labels, lr_test_pred, 'logistic regression')
plot_confusion_matrix(y_test_labels, rf_test_pred, 'random forest')
plot_confusion_matrix(y_test_labels, xgb_test_pred, 'XGBoost')
plot_confusion_matrix(y_test_temporal.argmax(axis=1), test_pred.argmax(axis=1), 'CNN-LSTM')

# learning curve for CNN-LSTM
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve, auc

def plot_roc_curves(y_true, y_pred_proba, model_name):
    """
    ROC curves for multiclass classification using predicted probabilities done by one vs rest (OVR)
    """
    plt.figure(figsize=(10, 8))
    auc_values = []
    stroke_types = list(dict.fromkeys(label_mapping.values()))
    
    colors = [
        '#FF0000',  # red
        '#00FF00',  # green
        '#0000FF',  # blue
        '#FFA500',  # orange
        '#800080',  # purple
        '#FFC0CB',  # pink
        '#008080',  # teal
        '#FFD700'   # gold
    ]
    
    for i, (class_name, color) in enumerate(zip(stroke_types, colors)):
        y_true_bin = (y_true == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)
        auc_values.append(roc_auc)
        
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})', color=color)
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title(f'ROC curves - {model_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    print(f'\naverage AUC for {model_name}: {np.mean(auc_values):.3f}')

# probability predictions (not class predictions)
lr_proba = lr.predict_proba(X_test_pca)   
rf_proba = best_rf.predict_proba(X_test_pca) 
xgb_proba = xgb.predict_proba(X_test_pca)
# already in prob
cnn_lstm_proba = test_pred  

plot_roc_curves(y_test_labels, lr_proba, 'logistic regression')
plot_roc_curves(y_test_labels, rf_proba, 'random forest')
plot_roc_curves(y_test_labels, xgb_proba, 'XGBoost')
plot_roc_curves(y_test_temporal.argmax(axis=1), cnn_lstm_proba, 'CNN-LSTM')