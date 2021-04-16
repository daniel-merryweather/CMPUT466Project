def main(filename="graphdata.txt"):
    file = open(filename, "r")
    result = file.read()
    file.close()

    result = result.split("\n")
    r = result[0]
    c = result[1]
    t = int(result[2])
    r = r.split(" ")
    c = c.split(" ")

    for i in range(len(r)):
        r[i] = int(r[i])
        c[i] = int(c[i])

    x = []
    yr = []
    yc = []
    for i in range(10):
        x.append((i+1)*5)
        r_total = 0
        c_total = 0
        for j in range(5):
            r_total += r[(i*5)+j]
            c_total += c[(i*5)+j]
        yr.append(r_total//5)
        yc.append(c_total/5)
    

    print("x = {}\nyr = {}\nyc = {}\nt = {}".format(x,yr,yc,t))

main()
