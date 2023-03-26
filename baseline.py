"""
CRF baseline for sequence tagging, essentially a copy of the pycrfsuite 
example.
"""

import argparse
import random
import pycrfsuite
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

def word2features(sent, i):
    word = sent[i][0]
    lowered_word = word.lower()
    features = [
        'bias',
        'word.lower=' + lowered_word,
        'word[-3:]=' + lowered_word[-3:],
        'word[-2:]=' + lowered_word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit()
    ]
    if i > 0:
        word1 = sent[i-1][0]
        lowered_word1 = word1.lower()
        features.extend([
            '-1:word.lower=' + lowered_word1,
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper()
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        lowered_word1 = word.lower()
        features.extend([
            '+1:word.lower=' + lowered_word1,
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper()
        ])
    else:
        features.append('EOS')
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for _, label in sent]

def sent2tokens(sent):
    return [token for token, _ in sent]

def load_data(task=1):
    with open(f"gua_spa_train_dev/task{task}/train.conllu") as f:
        sents = []
        for sent in f.read().strip("\n").split("\n\n"):
            sent = [
                line.strip("\t \n").split("\t") 
                for line in sent.split("\n") 
            ]
            sents.append(sent)

    random.shuffle(sents)
    train_size = int(len(sents) * 0.8)
    train = sents[:train_size]
    test = sents[train_size:]
    return train, test


def train_crf(X_train, y_train, model_name="task1_baseline.crfsuite"):
    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    trainer.train(model_name)

    tagger = pycrfsuite.Tagger()
    tagger.open(model_name)

    return tagger

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'0'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

def predict_and_evaluate(tagger, X_test, y_test):
    preds = [tagger.tag(feature_bundle) for feature_bundle in X_test]
    print(bio_classification_report(y_test, preds))
    return preds

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--task", default=1, help="1, 2, or 3")
    args = argparser.parse_args()

    train, test = load_data(task=args.task)
    
    X_train = [sent2features(s) for s in train]
    
    y_train = [sent2labels(s) for s in train]

    X_test = [sent2features(s) for s in test]
    y_test = [sent2labels(s) for s in test]

    model = train_crf(X_train, y_train)

    preds = predict_and_evaluate(model, X_test, y_test)

    dataforeval = []
    for i, sent in enumerate(test):
        for j, token in enumerate(sent):
            w = token[0]
            label = token[1]
            pred = preds[i][j]
            dataforeval.append(f"{w}\t{label}\t{pred}")
        dataforeval.append("")

    with open(f"baseline_predictions_task={args.task}", "w") as fout:
        fout.write("\n".join(dataforeval))
