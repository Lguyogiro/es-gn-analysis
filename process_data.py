
def write_separate_task_files(fpath, partition):
    with open(fpath) as f:
        lines = [line.strip("\n") for line in f]

    task1, task2, task3 = [], [], []

    for line in lines:
        if line.startswith("#"):
            continue

        elif not line.strip(" \n"):
            task1.append("")
            task2.append("")
            task3.append("")
            continue
        
        line = line.split("\t")
        if len(line) == 2:
            _, w = line
            task1.append(f"{w}\t0")
            task2.append(f"{w}\t0")
            task3.append(f"{w}\t0")
            continue

        else:
            _, word, tags = line
            if "-" in tags:
                t1, other = tags.split("-", 1)
                task1.append(f"{word}\t{t1}")
                if t1 == "ne":
                    task2.append(f"{word}\t{other}")
                    task3.append(f"{word}\t0")
                else:
                    task3.append(f"{word}\t{other}")
                    task2.append(f"{word}\t0")
            else:
                task1.append(f"{word}\t{tags}")
                task2.append(f"{word}\t0")
                task3.append(f"{word}\t0")

    import pdb;pdb.set_trace()

    with open(f"gua_spa_train_dev/task1/{partition}.conllu", "w") as f:
        f.write("\n".join(task1))
    with open(f"gua_spa_train_dev/task2/{partition}.conllu", "w") as f:
        f.write("\n".join(task2))
    with open(f"gua_spa_train_dev/task3/{partition}.conllu", "w") as f:
        f.write("\n".join(task3))

if __name__ == "__main__":
    trainpath = "gua_spa_train_dev/gua_spa_train.txt"
    devpath = "gua_spa_train_dev/gua_spa_dev.txt"
    write_separate_task_files(trainpath, "train")
    write_separate_task_files(devpath, "dev")

