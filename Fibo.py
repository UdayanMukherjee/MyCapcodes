def fib(i):
    if i<=1:
        return i
    else:
        return (fib(i-2)+fib(i-1))

n=int(input("Enter the number of consecutive Fibonacci numbers that need to be printed:"))
for i in range(n+1):
    print(fib(i), end=" ")
    