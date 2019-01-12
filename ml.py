from sklearn.datasets import load_iris
iris = load_iris()
print(list(iris.target_names))

from sklearn import tree
# we will create a decision tree classifier and assign it to variable so we can work with it, we will type classifier as the name of my variable, 
# and ww will set that to tree.DecisionTreeClassifier() that's function, 
classifier = tree.DecisionTreeClassifier()

# Now, we need to actually build the decisionTree through which  each new example will flow
# this decision tree can be built by feeding it both the training example and the target labels using the fit function like this
classifier = classifier.fit(iris.data, iris.target)
#so just review, we created a decision tree model and 
#now we are building the decision tree using its fit function which takes a set of example and the target labesl
#at this point the data is loaded and we have built a decision tree based on that data
#now finally comes the part we have been working toward, making predictions.
# we can do this using the prediction funtion on the decision tree classifier,like this
print(classifier.predict([[5.1,3.5,1.4,1.5]]))