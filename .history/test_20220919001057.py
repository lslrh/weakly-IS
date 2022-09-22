n,m = 3,30
inp = [2,1,3]

cur = sorted(inp)
while m:
    cur[0]+=1
    m -= 1
    i = 1
    flag = False
    while i<n:
        if cur[0] > cur[i]:
            i += 1
            flag = True
            break
        else:
            i += 1
        

    if flag:
        val = cur.pop(0)
        cur.insert(i, val)

print(cur[0])

