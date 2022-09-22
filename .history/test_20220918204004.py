n,m = 6,3
inp = [2,1,3,0,1,1]

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
        
    if flag:
        val = cur.pop(0)
        cur.insert(i, val)

print(cur[0])

