import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
import matplotlib.pyplot as plt


file_path = "data.xlsx"
df = pd.read_excel(file_path)
X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values  

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, clf in classifiers.items():
    acc_list, prec_list, rec_list, f1_list, auc_list = [], [], [], [], []
    all_y_true = []
    all_y_prob = []
    for train_idx, test_idx in skf.split(X_scaled, y_encoded):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)

        acc_list.append(accuracy_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred, average='macro'))
        rec_list.append(recall_score(y_test, y_pred, average='macro'))
        f1_list.append(f1_score(y_test, y_pred, average='macro'))
        # 多分类AUC
        y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
        auc = roc_auc_score(y_test_binarized, y_prob, average='macro', multi_class='ovr')
        auc_list.append(auc)

    print(f"=== {name} ===")
    print(f"Accuracy:  {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"Precision: {np.mean(prec_list):.4f} ± {np.std(prec_list):.4f}")
    print(f"Recall:    {np.mean(rec_list):.4f} ± {np.std(rec_list):.4f}")
    print(f"F1-score:  {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print(f"AUC:       {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}\n")

    all_y_true_bin = label_binarize(all_y_true, classes=[0, 1, 2])
    all_y_prob = np.array(all_y_prob)
    fpr = dict()
    tpr = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(all_y_true_bin[:, i], all_y_prob[:, i])
        plt.plot(fpr[i], tpr[i], label=f"Class {le.inverse_transform([i])[0]}")
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.show()