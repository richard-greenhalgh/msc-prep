# the "birthday problem", given a number of people n in the same room, what is the probability at least 2 share the same birthday?

exp = 10000
n = 23
data = matrix(sample(1:365, n*exp, replace=TRUE), nrow=n, ncol=exp)

# duplicated(x) : returns vector of TRUE/FALSE based on if corresponing element is NOT UNIQUE
# any(x) : returns TRUE if any of the elements are TRUE
# apply(x, dim, ...) apply a function to row (dim=1) or col (dim=2)

has_match <- apply(data, 2, function(x) any(duplicated(x)))
# has_match is vector of experiments, with TRUE (had a match) or FALSE (no match)

# calculate probability
ans = sum(has_match) / length(has_match)
cat("birthday problem with [", n, "] people and [", exp, "] experiments: ", 100*ans, "%\n")

