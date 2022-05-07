import random


def write_output(args, raw_res):
    with open(args.output_file, "w") as w:
        with open(args.test_file) as f:
        #with open('/home/rong/work/NER/homework/shouxie/data/hr_dev_test.txt') as f:
            for line, res_ in zip(f, raw_res):
                line = line.strip()
                for i, ch in enumerate(line):
                    if i == len(line)-1:
                        w.write(ch + "  ")
                        break
                    else:
                        #后面跟着B或S
                        if res_[i+1] == 0 or res_[i+1] == 3:
                            w.write(ch + "  ")
                        elif res_[i] == 4 and random.randint(0,1) == 0:
                            w.write(ch + "  ")
                        else:
                            w.write(ch)
                w.write('\n')
