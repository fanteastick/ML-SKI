def grades_sum(scores):
	total = 0
	for i in scores:
		total+=i
	return total



def otherGrades(scores):
	total=0
	i=0
	while i<len(scores):
		total+=scores[i]
		i+=1
	return total

print(otherGrades([1, 2, 3, 4, 5, 6, 7, 8, 9]))
print (1+2+3+4+5+6+7+8+9)


print(grades_sum([1, 2, 3, 4, 5, 6, 7, 8, 9]))
print (1+2+3+4+5+6+7+8+9)