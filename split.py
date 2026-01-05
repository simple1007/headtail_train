mpf = open("mps.txt","w",encoding="utf-8")
tagf = open("tags.txt","w",encoding="utf-8")
with open("C:\\Users\\ty341\\Desktop\\headtail_train\\dataset\\ht_dataset\\modu\\train\\headtail.txt",encoding="utf-8") as f:
    htsep="hththththt"
    tagsep="@@@"

    for l in f:
        l = l.strip()
        mp = []
        tag = []

        for ll in l.split():
            ll = ll.split(htsep)
            # print(ll)
            mtmp =[]
            ttmp = []
            for lll in ll:
                mp_tag=lll.split(tagsep)
                mtmp.append(mp_tag[0])
                ttmp.append(mp_tag[1])
                # for llll in mp_tag:
            mp.append(htsep.join(mtmp))
            tag.append(htsep.join(ttmp))
        # print(mp)
        # print(tag)
        if len(mp) > 5:
            mpf.write(" ".join(mp)+"\n")
            tagf.write(" ".join(tag)+"\n")
