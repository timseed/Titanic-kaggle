# coding: utf-8
from KagTitan import *

#
# If you want to run this on its own then.... this should work
#

#
#Why I have done it like this.... ?
#You should be able to inherit KagTitan, and then by overloading the LoadCSV module
#You can create new modules easily

#
#Why Create new modules ? 
#So you can easily see if they are an improvement


if __name__=="__main__":
	ShipSink = KagTitan()
	tr = ShipSink.LoadCsvFile('train.csv')
	ts = ShipSink.LoadCsvFile('test.csv')


	# Machine Learning time
	#
	# Import the random forest package
	# Create the random forest object which will include all the parameters
	# for the fit
	forest = RandomForestClassifier(n_estimators=1000, verbose=1, n_jobs=2)
	# Fit the training data to the Survived labels and create the decision trees
	fittedforest = forest.fit(tr[0::, 1::], tr[0::, 0])
	# Take the same decision trees and run it on the test data
	output = fittedforest.predict(ts).astype(int)
	pprint(fittedforest)

	print(output)
	np.savetxt('predict', output, delimiter=",")
	#open_file_object = csv.writer(predictions_file)
	#open_file_object.writerow(["PassengerId", "Survived"])
	#open_file_object.writerows(list(zip(ids, output)))
	# predictions_file.close()


	#
	# Try and see what is being used
	#
	importances = fittedforest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in fittedforest.estimators_],
		     axis=0)
	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print("Feature ranking:")
	fcnt = 16
	for f in range(fcnt):
	    print(("%d. feature %d (%f)" %
		   (f + 1, indices[f], importances[indices[f]])))

	# Plot the feature importances of the forest
	plt.figure()
	plt.title("Feature importances")
	plt.bar(list(range(fcnt)), importances[indices],
		color="r", yerr=std[indices], align="center")
	plt.xticks(list(range(fcnt)), indices)
	plt.xlim([-1, fcnt])

	print("Fitted Forest score is %f" %
	      (100.0 * fittedforest.score(tr[0::, 1::], tr[0::, 0])))
	plt.show()
