import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import heapq
import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential, Model
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt

K = 5

def create_ds(df0, df1):
    anchors = []
    pos = []
    neg = []
    labels = []

    for i in range(len(df0)):
        max_dist = []
        min_dist = []
        temp_pos = []
        temp_neg = []
        labels.append(0)
        anchors.append(df0[i])
        for j in range(len(df0)):
            dist = np.linalg.norm(df0[i] - df0[j])
            max_dist.append(dist)
        max_idx = heapq.nlargest(K, range(len(max_dist)), max_dist.__getitem__)
        for idx in max_idx:
            temp_pos.append(df0[idx])
        pos.append(np.array(temp_pos))

        for j in range(len(df1)):
            dist = np.linalg.norm(df0[i] - df1[j])
            min_dist.append(dist)
        min_idx = heapq.nsmallest(K, range(len(min_dist)), min_dist.__getitem__)
        for idx in min_idx:
            temp_neg.append(df1[idx])
        neg.append(np.array(temp_neg))

    for i in range(len(df1)):
        max_dist = []
        min_dist = []
        temp_pos = []
        temp_neg = []
        labels.append(1)
        anchors.append(df1[i])
        for j in range(len(df1)):
            dist = np.linalg.norm(df1[i] - df1[j])
            max_dist.append(dist)
        max_idx = heapq.nlargest(K, range(len(max_dist)), max_dist.__getitem__)
        for idx in max_idx:
            temp_pos.append(df1[idx])
        pos.append(np.array(temp_pos))
        for j in range(len(df0)):
            dist = np.linalg.norm(df1[i] - df0[j])
            min_dist.append(dist)
        min_idx = heapq.nsmallest(K, range(len(min_dist)), min_dist.__getitem__)
        for idx in min_idx:
            temp_neg.append(df0[idx])
        neg.append(np.array(temp_neg))

    return (np.array(anchors), np.squeeze(np.array(pos)), np.squeeze(np.array(neg))), np.array(labels)


def create_model():
    inputs = tf.keras.Input(shape=(23,), name="digits")
    x1 = tf.keras.layers.Dense(16, activation="relu", name="dense_1")(inputs)
    encoder = tf.keras.Model(inputs=inputs, outputs=x1)

    classifier = Sequential(
        [Input(shape=(16,)), Dense(1, activation='sigmoid')]
    )
    
    return encoder, classifier


def train_and_evaluate(lamda=0.5, temperature=0.5):
    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.5, gamma=2)
    acc = tf.keras.metrics.BinaryAccuracy()
    val_acc = tf.keras.metrics.BinaryAccuracy()
    optimizer = tf.keras.optimizers.Adam()
    EPOCHS = 300
    
    encoder, classifier = create_model()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=115)
    data = pd.read_csv('./data/all_data.csv', header=None)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_class0 = data[data.iloc[:, 23] == 0].reset_index(drop=True)
    data_class1 = data[data.iloc[:, 23] == 1].reset_index(drop=True)
    scaled_class0 = scaler.fit_transform(data_class0.iloc[:, 0:23])
    scaled_class1 = scaler.fit_transform(data_class1.iloc[:, 0:23])

    X, y = create_ds(scaled_class0, scaled_class1)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    for fold, (train_index, test_index) in enumerate(skf.split(X[0], y)):
        anchor, pos, neg = X[0], X[1], X[2]
        anchor_trn, anchor_val = anchor[train_index], anchor[test_index]
        pos_trn, pos_val = pos[train_index], pos[test_index]
        neg_trn, neg_val = neg[train_index], neg[test_index]
        labels_trn, labels_val = y[train_index], y[test_index]

        anchors = tf.data.Dataset.from_tensor_slices(anchor_trn)
        pos = tf.data.Dataset.from_tensor_slices(pos_trn)
        neg = tf.data.Dataset.from_tensor_slices(neg_trn)
        labels = tf.data.Dataset.from_tensor_slices(labels_trn)
        dataset = tf.data.Dataset.zip(((anchors, pos, neg), labels)).shuffle(1000, seed=16).batch(64)

        anchors_val = tf.data.Dataset.from_tensor_slices(anchor_val)
        pos_val = tf.data.Dataset.from_tensor_slices(pos_val)
        neg_val = tf.data.Dataset.from_tensor_slices(neg_val)
        labels_val = tf.data.Dataset.from_tensor_slices(labels_val)
        dataset_val = tf.data.Dataset.zip(((anchors_val, pos_val, neg_val), labels_val)).batch(64)

        for epoch in range(EPOCHS):
            start = time.time()

            for (batch, ((anchor, pos, neg), label)) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    cl_loss = 0.0
                    total = 0.0
                    tf.cast(label, dtype=tf.int64)
                    anchor = tf.cast(anchor, dtype=tf.float32)
                    pos = tf.cast(pos, dtype=tf.float32)
                    neg = tf.cast(neg, dtype=tf.float32)
                    anchor_f = encoder(anchor)
                    logits = classifier(anchor_f)
                    zp = tf.math.l2_normalize(anchor_f, axis=1)
                    
                    for i in range(pos.shape[1]):
                        pos_f = encoder(pos[:, i, :])
                        zi_pos = tf.math.l2_normalize(pos_f, axis=1)
                        sim = tf.linalg.diag_part(tf.matmul(zp, zi_pos, transpose_b=True))
                        total += tf.math.exp(sim / temperature)
                        
                    for i in range(neg.shape[1]):
                        neg_f = encoder(neg[:, i, :])
                        zi_neg = tf.math.l2_normalize(neg_f, axis=1)
                        sim = tf.linalg.diag_part(tf.matmul(zp, zi_neg, transpose_b=True))
                        total += tf.math.exp(sim / temperature)
                        
                    for i in range(pos.shape[1]):
                        pos_f = encoder(pos[:, i, :])
                        zi_pos = tf.math.l2_normalize(pos_f, axis=1)
                        sim = tf.linalg.diag_part(tf.matmul(zp, zi_pos, transpose_b=True))
                        nominator = tf.math.exp(sim / temperature)
                        step_loss = -tf.math.log(nominator / total)
                        step_loss = tf.reduce_sum(step_loss) / nominator.shape[0]
                        cl_loss += step_loss

                    total_cl_loss = cl_loss / pos.shape[1]
                    loss_value = loss_fn(label, logits)
                    loss = lamda * cl_loss + loss_value
                    
                grads = tape.gradient(loss, encoder.trainable_weights + classifier.trainable_weights)
                optimizer.apply_gradients(zip(grads, encoder.trainable_weights + classifier.trainable_weights))
                acc.update_state(label, logits)

                if batch % 4 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        epoch + 1, batch, loss_value + lamda * total_cl_loss))

            train_acc = acc.result()
            print("Fold: %.1d Training acc over epoch : %.4f" % (fold, train_acc))
            acc.reset_state()

            for (batch, ((anchor, pos, neg), label)) in enumerate(dataset_val):
                anchor_f_val = encoder(anchor, training=False)
                logits_val = classifier(anchor_f_val, training=False)
                val_acc.update_state(label, logits_val)

            validation_acc = val_acc.result()
            val_acc.reset_state()
            print("Fold: %.1d val acc: %.4f" % (fold, validation_acc))
            print("time taken : %.2fs" % (time.time() - start))

        y_score = []
        y_true = []
        y_pred = []

        ds_test = dataset_val.unbatch()
        for (anchor, pos, neg), label in ds_test:
            anchor = tf.expand_dims(anchor, 0)
            anchor_f_val = encoder(anchor, training=False)
            logits_val = classifier(anchor_f_val, training=False)
            if logits_val > 0.50:
                y_pred.append(1)
            else:
                y_pred.append(0)
            y_true.append(label)
            y_score.append(np.squeeze(logits_val))

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        auc = metrics.auc(fpr, tpr)
        print("fold {} auc value : {}".format(fold, auc))
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        f1score = metrics.f1_score(y_true, y_pred)
        print("fold {} precision value : {}".format(fold, precision))
        print("fold {} recall value : {}".format(fold, recall))
        print("fold {} f1score value : {}".format(fold, f1score))
        
        plt.figure(fold)
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


if __name__ == "__main__":
    train_and_evaluate(lamda=0.5, temperature=0.5)
