lst=[]
lst1=[]
lst = [int(i) for i in input("Enter the list items : ").split()]
for i in lst:
    if i>0:
        lst1.append(i)
print(lst1)
