import os
path = os.path.join(os.environ["DATASET"],"ht_dataset","train","headtail_train_mp.txt")
with open(path,encoding="utf-8") as mps:
    # for m,p in zip(morphs,tags):
    for mp in mps:
        for mpp_ in mp.split():
            flag = False
            for mpp in mpp_.split("+"):
                if ("NNG" in mpp or "NNP" in mpp) and "_J" in mpp:
                    # for mm, pp in zip(m.strip().split(),p.strip().split()):
                    #     print(mm,"/",pp,end=" ")
                    # print()
                    flag = True
                    print(mpp)
                    print(mp)
                    exit()
                    break
            if flag:
                break
            