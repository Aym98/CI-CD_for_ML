import pandas as pd
import matplotlib.pyplot as plt

import skops.io as sio

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


RDM_SEED = 42
data_path = "Data/drug200.csv"


if __name__ == '__main__':
    drug_df = pd.read_csv(data_path)
    cat_cols = [1, 2, 3]
    num_cols = [0, 4]

    X = drug_df.drop("Drug", axis=1).values
    y = drug_df.Drug.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RDM_SEED)

    transform = ColumnTransformer(
        [
            ('encode', OrdinalEncoder(), cat_cols),
            ('num_impute', SimpleImputer(strategy='median'), num_cols),
            ('scale', StandardScaler(), num_cols)
        ]
    )

    pipe = Pipeline(
        steps=[
            ('transform', transform),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=RDM_SEED))
        ])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')

    print(f"Accuracy : {round(acc, 4) * 100}, F1 : {round(f1, 4) * 100}")

    res_path = 'Results/results.txt'
    with open(res_path, 'w') as f:
        f.write(f"Accuracy : {round(acc, 4) * 100}, \nF1 : {round(f1, 4) * 100}")

    cm = confusion_matrix(y_test, preds, labels=pipe.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
    disp.plot()
    plt.savefig('Results/confusion_matrix.png')

    sio.dump(pipe, 'Model/pipeline.skops')
