# MIT 18.05 R Tutorial B

print("sample(x,k): k samples from vector x, NO REPLACEMENT (cannot pick the same element twice)")

x = 1:5
print(x)
for (i in 1:3) {
  y = sample(x,3)
  print(y)
}


print("sample(x,k,replace=TRUE): k samples from vector x, WITH REPLACEMENT")

x = 1:5
print(x)
for (i in 1:3) {
  y = sample(x,3,replace=TRUE)
  print(y)
}

print("generate 12 d6 rolls: y = sample(1:6, 12, replace=TRUE)")
y = sample(1:6, 12, replace=TRUE)
print(y)

print("pack into a 3x4 matrix: z = matrix(y, nrow=3, ncol=4)")
z = matrix(y, nrow=3, ncol=4)
print(z)

print("what % of 1000 d6 rolls yields a 6?  sum( sample(1:6, 1000, replace=TRUE) == 6 ) / 1000")
d = sample(1:6, 1000, replace=TRUE)
ans = sum(d == 6) / length(d)
print(ans)
cat("1/6 = ", 1/6, "\n")

print("probability of rolling at least one 6 in 4 rolls?")
n_experiment = 100
rolls_per = 4
total_rolls = n_experiment * rolls_per
data = matrix(sample(1:6, total_rolls, replace=TRUE), nrow=rolls_per, ncol=n_experiment)
#print(data)
print("sixes per 'experiment'")
count6 = colSums(data == 6)
print(count6)
ans = sum(count6 >= 1) / length(count6)
cat("probability = ", ans, "\n")

print("estimate probability of getting a total of 7 when rolling two d6")
n_experiment = 10000
rolls_per = 2
total_rolls = n_experiment * rolls_per
data = colSums( matrix(sample(1:6, total_rolls, replace=TRUE), nrow=rolls_per, ncol=n_experiment) )
print("use: data = colSums( matrix(sample(1:6, 2*N, replace=TRUE), nrow=2, ncol=N) )")
print("ans = sum(data == 7) / length(data)")
ans = sum(data == 7) / length(data)
cat("estimated probability = ", ans, "\n")



