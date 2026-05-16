
list = [1,2,3]

print(list)
for i in range(0,len(list)):
    print(list[i])


for i, value in enumerate(list):
    print(i, value)

print("==========")

print(list[1])
print(list[-1])

print(list[::])

print(list[:2])

print(list[0:0])

print(list[0:1])

print(list[1:2])

print("-------------")
# 列表推导式
list1 = [i ** 2 for i in range(10)]
print(list1)

list2 = [i * 2 for i in range(10) if i % 2 == 0]
print(list2)


list3 = [i * 3 for i in list]
print(list3)


list4 = [1,2,3]
list5 = ["a","b","c"]

list6 = [(i,j) for i in list for j in list]
print(list6)

# zip 拉链函数
list7 = zip(list1,list2)
for item in list7:
    print(item)