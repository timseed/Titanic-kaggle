# coding: utf-8
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
import pylab as plt
#get_ipython().magic('matplotlib inline')


class KagTitan(object):

    def __init__(self):
        self.Version = "1.0"

    def IsAdult(self, i):
        if i < 18:
            return 0
        return 1

    def atoi(self, s):
        try:
            return ord(s[0])
        except:
            return 0

    def LoadCsvFile(self, fn):
        #
        # Read DataFile
        #
        f = open(fn)
        train = pd.read_csv(f, header=0)

        # Modify Gender to 0,1 female/male
        train['Gender'] = train['Sex'].map(
            {'female': 0, 'male': 1}).astype(int)
        # Get Ages for Missing Values
        median_ages = np.zeros((2, 3))
        median_ages
        for i in range(0, 2):
            for j in range(0, 3):
                median_ages[i, j] = train[(train['Gender'] == i) &
                                          (train['Pclass'] == j + 1)]['Age'].dropna().mean()
        #
        # Add Family Present
        #
        train['FamilyGroup'] = 0

        train['CalcFare'] = train['Fare']
        avg_fare = np.zeros(3)
        for pc in range(0, 3):
            avg_fare[pc] = train[
                (train.Pclass == pc + 1)]['Fare'].dropna().mean()
        #
        # Now set prices
        for pc in range(0, 3):
            train.loc[(train.Fare.isnull()), 'CalcFare'] = avg_fare[pc]

        # Same for Fare and Gender

        af2 = np.zeros((2, 3))
        for i in range(0, 2):
            for j in range(0, 3):
                af2[i][j] = train[
                    (train['Gender'] == i) & (train['Pclass'] == j + 1)]['Fare'].dropna().mean()

        train['CalcFare2'] = train['Fare']

        for i in range(0, 2):
            for j in range(0, 3):
                train.loc[(train.Fare.isnull()) & (train.Gender == i) & (
                    train.Pclass == j + 1), 'CalcFare2'] = af2[i, j]

        # Add Deck of Cabin
        # A Deck 1
        # Z Deck 25
        #

        # Modify Age
        train['AgeFill'] = train['Age']
        for i in range(0, 2):
            for j in range(0, 3):
                train.loc[(train.Age.isnull()) & (train.Gender == i) & (train.Pclass == j + 1),
                          'AgeFill'] = median_ages[i, j]

        # Remove All String and Columns with Null in them
        #train = train.drop(['PassengerId','Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Age'], axis=1)
        train_exp = pd.DataFrame()
        try:
            train_exp['Survived'] = train['Survived']
        except:
            pass
         # train_exp['AgeFill']=train['AgeFill']
        train_exp['Gender'] = train['Gender']
        train_exp['Pclass'] = train['Pclass']
        train_exp['Adult'] = train['AgeFill'].map(lambda x: self.IsAdult(x))

        train_exp['1AF'] = 0
        train_exp.loc[(train.Gender == 0) & (
            train.AgeFill > 18) & (train.Pclass == 1), '1AF'] = 1
        train_exp['2AF'] = 0
        train_exp.loc[(train.Gender == 0) & (
            train.AgeFill > 18) & (train.Pclass == 2), '2AF'] = 1
        train_exp['3AF'] = 0
        train_exp.loc[(train.Gender == 0) & (
            train.AgeFill > 18) & (train.Pclass == 3), '3AF'] = 1

        # train_exp['1AM']=0
        # train_exp.loc[ (train.Gender == 1)& (train.AgeFill >18)&(train.Pclass==1),'1AM']=1
        train_exp['2AM'] = 0
        train_exp.loc[(train.Gender == 1) & (
            train.AgeFill > 18) & (train.Pclass == 2), '2AM'] = 1
        train_exp['3AM'] = 0
        train_exp.loc[(train.Gender == 1) & (
            train.AgeFill > 18) & (train.Pclass == 3), '3AM'] = 1

        train_exp['1WC'] = 0
        train_exp.loc[
            ((train.Gender == 0) | (train.AgeFill < 18)) & (train.Pclass == 1), '1WC'] = 1
        train_exp['2WC'] = 0
        train_exp.loc[
            ((train.Gender == 0) | (train.AgeFill < 18)) & (train.Pclass == 2), '2WC'] = 1
        train_exp['3WC'] = 0
        train_exp.loc[
            ((train.Gender == 0) | (train.AgeFill < 18)) & (train.Pclass == 3), '3WC'] = 1
        train_exp['CalcFare'] = train['CalcFare']
        train_exp['CalcFare2'] = train['CalcFare2']

        train_exp['1InFamily'] = 0
        train_exp.loc[(train.Pclass == 1) & (
            (train.Parch > 0) | (train.SibSp > 0)), '1InFamily'] = 1
        train_exp['2InFamily'] = 0
        train_exp.loc[(train.Pclass == 2) & (
            (train.Parch > 0) | (train.SibSp > 0)), '2InFamily'] = 1
        train_exp['3InFamily'] = 0
        train_exp.loc[(train.Pclass == 3) & (
            (train.Parch > 0) | (train.SibSp > 0)), '3InFamily'] = 1

        # Now Check the male passengers in case thet are below 18

    #  train_exp['Fare']=train['Fare']
        # Now Convert back to numpy
        # print train_exp.dtypes
        f.close()
        train_data = train_exp.values
        return train_data


#
# If you want to run this on its own then.... this should work
#

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
