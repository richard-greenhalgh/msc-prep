# 100,000 dice rolls

N = 100000
d6 = 1:6
rolls = sample(d6, N, replace=TRUE)

cat("mean of [", N, "] (d6) dice rolls: ", mean(rolls), "\n")

# =============================================================================

# run length encoding example: rle()
x = c(1,1,1,2,3,3,3,1,1)
y = rle(x)

cat("rle(  c(1,1,1,2,3,3,3,1,1)  ): \n\n")
print(y)

cat("\nuse e.g. y$lengths and y$values to access the results of rle()\n\n")

# =============================================================================

# binomial distribution and rle()

set.seed(10)
y = rbinom(20,1,.5)

l = rle(y)$lengths

cat("y = ")
print(y)
cat("longest run = ", max(l), "\n")


