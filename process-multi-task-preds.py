import sys
pred1, pred2, pred3 = sys.argv[1], sys.argv[2], sys.argv[3]

orig = "gua_spa_train_dev/gua_spa_dev.txt"
with open(orig) as f:
    sents = [line for line in f]

with open(pred3) as f:
    pred3_lines = [line for line in f]
with open(pred2) as f:
    pred2_lines = [line for line in f]
with open(pred1) as f:
    pred1_lines = [line for line in f]


for i, line in enumerate(pred1_lines):
    if line.startswith("#dev") or not line.strip("\n"):
        print(line.strip("\n"))
    else:
        id_ = sents[i].split("\t")[0]
        lid_ner_line = line.split("\t")
        lid_cs_line = pred2_lines[i].split("\t")
        lid_only_line = pred3_lines[i].split("\t")

        w = lid_ner_line[0]
        lang = lid_only_line[1]
        ner = lid_ner_line[2]
        cs = lid_cs_line[3].strip("\n")

        #w, tags = line.strip("\n").split("\t", 1)
        #lang, ner, cs = tags.split("\t")
        
        label = lang
        if lang == "ne" and ner != "O":
            label += "-" + ner.lower()
        elif lang == "es" and cs != "O":
            label += "-" + cs.lower()

        print(f"{id_}\t{w}\t{label}")

