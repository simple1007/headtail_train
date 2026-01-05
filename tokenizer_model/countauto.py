with open('httk_x.txt',encoding='utf-8') as f:
    with open('httk_y.txt',encoding='utf-8') as ff:
        count = 0
        for l, ll in zip(f,ff):
            if len(l) != len(ll):
                count += 1
                print(l)
                print(ll)
                print(len(l),len(ll))
                for l_,ll_ in zip(l,ll):
                    print(l_,ll_)
                print(ll[-1])
                exit()

print(count)
