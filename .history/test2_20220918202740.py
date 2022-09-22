n = 3

inp = [[1, 10, 10], [3, 4, 11], [4, 5, 3]]
print(inp)
inp = sorted(inp)
print(inp)
max_v = 0
for i in range(n):
    start, end, value = inp[i]
    cur_v = value
    for inter in range(i+1, n):
        s, e, v = inp[inter]
        if s<end:
            continue
        else:
            cur_v += v
    max_v = max(max_v, cur_v)
print(max_v)