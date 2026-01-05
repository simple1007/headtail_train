import os
def noramlize_line(datapath:str):
    htsep = "hththththt"

    resx = open(os.path.join(datapath,"kiwimorphs_.txt"),"w",encoding="utf-8")
    resy = open(os.path.join(datapath,"kiwitags_.txt"),"w",encoding="utf-8")
    with open(os.path.join(datapath,"delkiwimorphs.txt"),encoding="utf-8") as m , open(os.path.join(datapath,"delkiwitags.txt"),encoding="utf-8") as t:
        # print(count)
        for mm,tt in zip(m,t):
            l = mm#in_f.readline()+' '
            # ltt = l 
            #l = l.strip()
            l = l.replace("+","_")
            import copy
            ltmp = []#copy.deepcopy(l)
            for l_ in l.split():
                l_ = l_.strip(htsep)
                ltmp.append(l_)
            l = " ".join(ltmp)+" "
            l = l.replace(htsep,"+")

            l = l.strip()
            if len(l.split()) <= 3:
                continue
            
            resx.write(mm)
            resy.write(tt)
    resx.close()
    resy.close()