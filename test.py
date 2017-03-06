

if __name__=="__main__":
    init_v1 = 1
    init_v2 = 1
    n = 4
    for i in range(0,n):
        temp = init_v2
        init_v2 = init_v1 + init_v2
        init_v1 = temp
        print("Iniv 1: " + str(init_v1) + ", Init_v2: " + str(init_v2))
    print(str(init_v2))