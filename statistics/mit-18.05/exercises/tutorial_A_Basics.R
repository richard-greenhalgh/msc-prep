# explore basic R syntax

x <- 3
print("x <- 3")
print(x)

x <- c(1.0, 1.5, 2.5)
print("x <- c(1.0, 1.5, 2.5)")
print(x)

y <- c(2.0, 4.0, 1.0)
z = x * y
print("y <- c(2.0, 4.0, 1.0)")
print("z = x * y")
print(z)

x <- 1:4
print("x <- 1:4")
print(x)

y <- x[ c(2,3) ]
print("y <- x[ c(2,3) ]")
print(y)

print("functions/constants: sin() / exp() / sum() / mean() /  pi")

x <- 1:12
print("x <- 1:12")
y = matrix(x, nrow=3, ncol=4, byrow=FALSE)
print("y = matrix(x, nrow=3, ncol=4, byrow=FALSE)")
print(y)

y = matrix(x, nrow=4, ncol=3, byrow=TRUE)
print("y = matrix(x, nrow=4, ncol=3, byrow=TRUE)")
print(y)

z = colSums(y * 1.0)
print("z = colSums(y)")
print(z)

z = rowMeans(y * 1.0)
print("z = rowMeans(y)")
print(z)

