"""
CRF baseline for sequence tagging, essentially a copy of the pycrfsuite 
example.
"""

import argparse
import random
import pycrfsuite
from emoji import UNICODE_EMOJI
from string import punctuation
from itertools import chain
import langid
from nltk import ngrams
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer


def override_preds(token_sents, preds, task=1):
    if task == 1:
        new_preds = []
        for j, sent in enumerate(token_sents):
            new_sent_preds = []
            for i, w in enumerate(sent):
                if all((ch in UNICODE_EMOJI["en"] or ch in punctuation or ch in "0123456789") for ch in w):
                    new_sent_preds.append("other")
                else:
                    new_sent_preds.append(preds[j][i])
            new_preds.append(new_sent_preds)
        return new_preds
    elif task == 3:
        new_preds = []
        for j, sent in enumerate(token_sents):
            new_sent_preds = []
            prev_was_cc = False
            for i, w in enumerate(sent):
                lang_pred, score = langid.classify(w)
                if lang_pred == "es":
                    if prev_was_cc is False:
                        new_sent_preds.append("b-cc")
                        prev_was_cc = True
                    else:
                        new_sent_preds.append("i-cc")
                else:
                    new_sent_preds.append(preds[j][i])
                    prev_was_cc = False
            new_preds.append(new_sent_preds)
        return new_preds
    else:
        return preds
        
def word2features(sent, i):
    word = sent[i][0]
    lowered_word = word.lower()
    bigrams = ["".join(bg) for bg in ngrams("^" + lowered_word + "$", 2)]
    features = [
        'bias',
        'word.lower=' + lowered_word,
        'word[-3:]=' + lowered_word[-3:],
        'word[-2:]=' + lowered_word[-2:],
        'word[:2]=' + lowered_word[:2],
        'word[:3]=' + lowered_word[:3],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit()
    ]
    features.extend([f"bigram={bg}" for bg in bigrams])

    if i > 0:
        word1 = sent[i-1][0]
        lowered_word1 = word1.lower()
        features.extend([
            '-1:word.lower=' + lowered_word1,
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word[-3:]=' + lowered_word1[-3:],
            '-1:word[-2:]=' + lowered_word1[-2:],
            '-1:word[:2]=' + lowered_word1[:2],
            '-1:word[:3]=' + lowered_word1[:3],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        lowered_word1 = word.lower()
        features.extend([
            '+1:word.lower=' + lowered_word1,
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1word[-3:]=' + lowered_word1[-3:],
            '+1word[-2:]=' + lowered_word1[-2:],
            '+1word[:2]=' + lowered_word1[:2],
            '+1word[:3]=' + lowered_word1[:3],
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

def load_data(train_only=None, dev=True):
    if dev is True:
        p = f"official_data/dev_combined.conllu"
    else:
        p = f"official_data/train_combined.conllu"

    with open(p) as f:
        sents = []
        for sent in f.read().strip("\n").split("\n\n"):
            sent = [
                line.strip("\t \n").split("\t") 
                for line in sent.split("\n") 
            ]

            sents.append(sent)

    if train_only is not None or dev is True:
        return sents
    
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
        
    tagset = set(lb.classes_) - {'O', '0'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

def evaluate(labels, preds):
    print(bio_classification_report(labels, preds))

def predict(tagger, test_tokens, X_test, task):
    preds = [tagger.tag(feature_bundle) for feature_bundle in X_test]
    preds = override_preds(test_tokens, preds, task=task)
    return preds

def predict_and_evaluate(tagger, test_tokens, X_test, y_test, task):
    preds = predict(tagger, test_tokens, X_test, task)
    evaluate(y_test, preds)
    return preds


def get_nfold_data(n_folds, task=1):
    with open(f"gua_spa_train_dev/task{task}/train.conllu") as f:
        sents = []
        for sent in f.read().strip("\n").split("\n\n"):
            sent = [
                line.strip("\t \n").split("\t") 
                for line in sent.split("\n") 
            ]
            sents.append(sent)
    random.shuffle(sents)
    test_size = len(sents) // n_folds
    folded = []
    idx = 0
    for _ in range(n_folds):
        folded.append(sents[idx:idx+test_size])
        idx += test_size
    for i in range(n_folds):
        test = folded[i]
        train = [sent for j in range(len(folded)) 
                 for sent in folded[j] if j != i]
        yield train, test

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--task", default=1, help="1, 2, or 3")
    argparser.add_argument("--xval", action="store_true")
    argparser.add_argument("--train_only", action="store_true")
    argparser.add_argument("--predict", action="store_true")
    argparser.add_argument("--model")
    argparser.add_argument("--path_to_data")
    argparser.add_argument("--data_to_predict")
    argparser.add_argument("--nfolds", default=10)
    argparser.add_argument("--save_model_path")

    args = argparser.parse_args()

    if args.xval:
        all_predictions = []
        all_labels = []
        fold_iter = get_nfold_data(args.nfolds, task=args.task)
        for i, (train, test) in enumerate(fold_iter):
            X_train = [sent2features(s) for s in train]
            y_train = [sent2labels(s) for s in train]
            X_test = [sent2features(s) for s in test]
            y_test = [sent2labels(s) for s in test]
            test_tokens = [sent2tokens(s) for s in test]
            model = train_crf(X_train, y_train, model_name=f"task{args.task}_baseline.crfsuite")
            preds = predict(model, test_tokens, X_test, args.task)
            all_predictions.extend(preds)
            all_labels.extend(y_test)
            print(f"Finished fold {i}")
        evaluate(all_labels, all_predictions)
    
    elif args.train_only:
        train = load_data(dev=False)[0]
        import pdb;pdb.set_trace()
        X_train = [sent2features(s[0]) for s in train]
        y_train = [sent2labels(s) for s in train]
        model = train_crf(X_train, y_train, model_name=f"task{args.task}_baseline.crfsuite")


    elif args.predict:
        tagger = pycrfsuite.Tagger()
        tagger.open(args.model)
        dev = load_data(task=args.task, dev=True)
        X_test = [sent2features(s) for s in dev]
        test_tokens = [sent2tokens(s) for s in dev]

        preds = predict(tagger, 
                        test_tokens,
                        X_test, 
                        args.task)
        with open(f"preds_task{args.task}.txt", "w") as f:
            f.write("\n\n".join(["\n".join(s) for s in preds]))


    else:
        train, test = load_data(task=args.task)
        X_train = [sent2features(s) for s in train]
        y_train = [sent2labels(s) for s in train]
        X_test = [sent2features(s) for s in test]
        y_test = [sent2labels(s) for s in test]
        test_tokens = [sent2tokens(s) for s in test]
        model = train_crf(X_train, y_train)

        preds = predict_and_evaluate(model, 
                                     test_tokens, 
                                     X_test, 
                                     y_test, 
                                     args.task)
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
    
