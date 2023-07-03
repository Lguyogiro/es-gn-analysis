
def write_separate_task_files(fpath, partition):
    with open(fpath) as f:
        lines = [line.strip("\n") for line in f]

    task1, task2, task3 = [], [], []
    combined = []
    for line in lines:
        if line.startswith("#"):
            combined.append(line)
            continue

        elif not line.strip(" \n"):
            task1.append("")
            task2.append("")
            task3.append("")
            combined.append("")
            continue
        
        line = line.split("\t")
        if len(line) == 2:
            _, w = line
            task1.append(f"{w}\t0")
            task2.append(f"{w}\t0")
            task3.append(f"{w}\t0")
            combined.append(f"{w}\t_\t_\t_")
            continue

        else:
            _, word, tags = line
            if "-" in tags:
                t1, other = tags.split("-", 1)
                task1.append(f"{word}\t{t1}")
                if t1 == "ne":
                    task2.append(f"{word}\t{other}")
                    task3.append(f"{word}\t0")
                    combined.append(f"{word}\t{t1}\t{other}\t0")
                else:
                    task3.append(f"{word}\t{other}")
                    task2.append(f"{word}\t0")
                    combined.append(f"{word}\t{t1}\t0\t{other}")
            else:
                task1.append(f"{word}\t{tags}")
                task2.append(f"{word}\t0")
                task3.append(f"{word}\t0")
                combined.append(f"{word}\t{tags}\t0\t0")


    # with open(f"gua_spa_train_dev/task1/{partition}.conllu", "w") as f:
    #     f.write("\n".join(task1))
    # with open(f"gua_spa_train_dev/task2/{partition}.conllu", "w") as f:
    #     f.write("\n".join(task2))
    # with open(f"gua_spa_train_dev/task3/{partition}.conllu", "w") as f:
    #     f.write("\n".join(task3))

    with open(f"gua_spa_train_dev_test/{partition}_combined.conllu", "w") as f:
        f.write("\n".join(combined))

if __name__ == "__main__":
    trainpath = "gua_spa_train_dev_test/gua_spa_train.txt"
    devpath = "gua_spa_train_dev_test/gua_spa_dev_gold.txt"
    testpath = "gua_spa_train_dev_test/gua_spa_test.txt"

    write_separate_task_files(trainpath, "train")
    write_separate_task_files(devpath, "dev")
    write_separate_task_files(testpath, "test")

