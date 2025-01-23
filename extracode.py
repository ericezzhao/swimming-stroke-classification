# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:40:03 2024

@author: zhaoez
"""

'''lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
results['logistic regression'] = classification_report(y_test, lr_pred, digits = 6)
y_train_pred = lr.predict(X_train)
lr_train_accuracy = (y_train_pred == y_train).mean()
print('lr train accuracy: ', lr_train_accuracy)'''

'''rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
results['random forest'] = classification_report(y_test, rf_pred, digits = 6)
y_train_pred = rf.predict(X_train)
rf_train_accuracy = (y_train_pred == y_train).mean()
print('rf train accuracy: ', rf_train_accuracy)'''

'''
# rf
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', 
                          max_depth=6, min_samples_leaf=10,
                          class_weight=class_weight_dict, 
                          bootstrap=True, random_state=42)
rf.fit(X_train_balanced, y_train_balanced)
rf_train_pred = rf.predict(X_train_balanced)
rf_test_pred = rf.predict(X_test_pca)
print("\nrf training:")
print(classification_report(y_train_balanced, rf_train_pred, digits=6))
'''

'''xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
results['XGBoost'] = classification_report(y_test, xgb_pred, digits = 6)
y_train_pred = xgb.predict(X_train)
xgb_train_accuracy = (y_train_pred == y_train).mean()
print('xgb train accuracy: ', xgb_train_accuracy)'''

'''print("\nCross-Validation Results:")
for model_name, scores in results.items():
    print(f"\n{model_name}:")
    print("training:", scores['train_score'])
    print(f"training mean: {scores['train_score'].mean():.4f} (+/- {scores['train_score'].std() * 2:.4f})")
    print("\nCV scores:", scores['test_score'])
    print(f"mean CV score: {scores['test_score'].mean():.4f} (+/- {scores['test_score'].std() * 2:.4f})")'''
'''for model_name, report in results.items():
    print(f"\n{model_name} results:")
    print(report)'''
    
# lstm
'''def create_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model'''
'''def create_cnn_lstm_model(input_shape, num_classes):
    model = Sequential([
        # CNN layers
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        
        # lstm layers
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

X = X_resampled
y = y_resampled_onehot
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("training shape:", X_train.shape)
print("test shape:", X_test.shape)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
cnn_lstm_model = create_cnn_lstm_model((X_train.shape[1], X_train.shape[2]), num_classes=8)
#lstm_model = create_lstm_model(input_shape, num_classes)lstm_model((X_train.shape[1], X_train.shape[2]), num_classes=8)

cnn_lstm_model.summary()

history = cnn_lstm_model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping])

cnn_lstm_pred = cnn_lstm_model.predict(X_test)
print("\nCNN-LSTM results:")
print(classification_report(y_test.argmax(axis=1), cnn_lstm_pred.argmax(axis=1), digits=6))'''

''' 
# SMOTE
# onehot -> single column
y_train_labels = np.argmax(y_train, axis=1)
print("original class distribution:")
print(Counter(y_train_labels))
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_pca, y_train_labels)
y_train_smote_onehot = to_categorical(y_train_smote)
print("resampled class distribution:")
print(Counter(y_train_smote))

print("\noriginal shapes:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print("\nresampled shapes:")
print(f"X_resampled: {X_train_smote.shape}")
print(f"y_resampled: {y_train_smote_onehot.shape}")

# compare SMOTE impact on distribution
plt.figure(figsize=(15, 5))
# before SMOTE
plt.subplot(1, 2, 1)
counts_before = Counter(y_train_labels)
plt.bar(counts_before.keys(), counts_before.values())
plt.title("class distribution before SMOTE")
plt.xlabel("class")
plt.ylabel("count")
# invidual class scores (F1)
# after SMOTE
plt.subplot(1, 2, 2)
counts_after = Counter(y_train_smote)
plt.bar(counts_after.keys(), counts_after.values())
plt.title("class distribution after SMOTE")
plt.xlabel("class")
plt.ylabel("count")
plt.tight_layout()
plt.show()'''

'''
# SMOTE for training data
X_temporal = data_array.reshape(data_array.shape[0], -1, 9)
X_train_temporal, X_test_temporal, y_train_temporal, y_test_temporal = train_test_split(X_temporal, labels_one_hot, test_size=0.2, random_state=42)
X_train_temporal_flat = X_train_temporal.reshape(X_train_temporal.shape[0], -1)
y_train_temporal_labels = np.argmax(y_train_temporal, axis=1)
X_train_temporal_smote_flat, y_train_temporal_smote = smote.fit_resample(X_train_temporal_flat, y_train_temporal_labels)
X_train_temporal_smote = X_train_temporal_smote_flat.reshape(-1, X_train_temporal.shape[1], X_train_temporal.shape[2])
y_train_temporal_smote_onehot = to_categorical(y_train_temporal_smote)
'''