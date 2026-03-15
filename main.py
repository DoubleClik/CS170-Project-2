import pandas as pd
import random
import math ## math.dist() for euclidian distance

def readDatasetAndCreateDataframe():
    # Display dataset options to User
    print("1. CS170_Large_DataSet__42\n")
    print("2. CS170_Small_DataSet__5\n")
    print("3. SanityCheck_DataSet__1\n")
    print("4. SanityCheck_DataSet__2\n")

    # Prompt user to select a dataset option
    optionSelected = input("Choose DataSet: ")

    # Set file path to chosen dataset OR throw error if input is NOT 1, 2, 3, or 4
    if optionSelected == "1":
        filepath = f"datasets/CS170_Large_DataSet__42.txt"
    elif optionSelected == "2":
        filepath = f"datasets/CS170_Small_DataSet__5.txt"
    elif optionSelected == "3":
        filepath = f"datasets/SanityCheck_DataSet__1.txt"
    elif optionSelected == "4":
        filepath = f"datasets/SanityCheck_DataSet__2.txt"
    else:
        raise ValueError("Invalid opiton, please only enter [1, 2, 3, 4].")

    # Read dataset at filepath into dataframe
        # - Values are delimited by whitespace
    df = pd.read_csv(filepath, sep=r'\s+', header=None)

    return df

def leaveOneOutCrossValidation(df, currentSet, featureToAdd):
    dfNumRows = df.shape[0]
    successfulPredicitons = 0

    # Features to evaluate: current set plus the candidate feature
    featuresToUse = currentSet + [featureToAdd]

    # Outerloop to classify each row (object) in df
    for i, rowToClassify in enumerate(df.itertuples()):
        # itertuples: position 0 = pandas Index, position 1 = column 0 (class label), position N+1 = column N
        realClassification = rowToClassify[1]

        nearestNeighborDistance = float('inf')
        nearestNeighborLocation = -1
        nearestNeighborClass = -1

        # Coordinate of rowToClassify (x) to find euclidian distance
        xValue = [rowToClassify[feature + 1] for feature in featuresToUse]

        # Innerloop to find the nearest neighbor to row (object) being classified
        for j, rowBeingChecked in enumerate(df.itertuples()):
            # Do NOT check the same row as rowToClassify as a nearestNeighbor candidate
            if i != j:
                # Coordinate of rowBeingChecked (y) to find euclidian distance
                yValue = [rowBeingChecked[feature + 1] for feature in featuresToUse]

                # If distance between rowToClassify (x) and rowBeingChecked (y) is less than the previous best nearestNeighborDistance, then update
                distance = math.dist(xValue, yValue) # Python math libary dist() function for Euclidean distance
                if distance < nearestNeighborDistance:
                    nearestNeighborDistance = distance
                    nearestNeighborLocation = j
                    nearestNeighborClass = rowBeingChecked[1]

        # If class prediction is correct, increment succesfulPredicitons counter by 1
        if nearestNeighborClass == realClassification:
            successfulPredicitons += 1
        # print(f"Object {i} is Class {realClassification}")
        # print(f"Its nearest neighbor is Object {nearestNeighborLocation}, which is Class {nearestNeighborClass}")

    return successfulPredicitons / dfNumRows

def featureSelectionSearch(df):
    dfNumRows = df.shape[0]
    dfNumCols = df.shape[1] - 1

    # Candidate features not yet added to the current set
    features = list(range(1, dfNumCols + 1))
    # Ordered list of (feature, accuracy) as features are greedily selected
    currentFeaturesAndAccuracy = []

    bestOverallAccuracy = 0.0
    bestOverallSubset = []

    print(f"\nThis dataset has {dfNumCols} features (not including the class attribute), with {dfNumRows} instances.")

    # Accuracy with all features upfront
    allFeaturesAccuracy = leaveOneOutCrossValidation(df, list(range(1, dfNumCols)), dfNumCols)
    print(f'\nRunning nearest neighbor with all {dfNumCols} features, using "leaving-one-out" evaluation, I get an accuracy of {allFeaturesAccuracy * 100:.1f}%')

    print("\nBeginning Forward Selection search.\n")

    # Outer loop: each level greedily adds the single best remaining feature
    for level in range(1, dfNumCols + 1):
        featureIndex = 0
        bestIndex = 0
        bestAccuracy = 0.0
        bestFeature = -1

        # Features committed in previous levels (without accuracy values)
        currentSet = [f for f, _ in currentFeaturesAndAccuracy]

        # Inner loop: evaluate each remaining candidate by temporarily adding it
        for feature in features:
            accuracy = leaveOneOutCrossValidation(df, currentSet, feature)
            print(f"\tUsing feature(s) {set(currentSet + [feature])} accuracy is {accuracy * 100:.1f}%")

            # Track the candidate with the highest accuracy at this level
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                bestIndex = featureIndex
                bestFeature = feature
            featureIndex += 1

        # Commit the best candidate: move it from the pool into the current set
        item = features.pop(bestIndex)
        currentFeaturesAndAccuracy.append((item, bestAccuracy))
        currentSet = [f for f, _ in currentFeaturesAndAccuracy]

        # Warn if accuracy dropped, but continue searching in case of local maxima
        if bestAccuracy < bestOverallAccuracy:
            print(f"\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        else:
            bestOverallAccuracy = bestAccuracy
            bestOverallSubset = currentSet[:]

        print(f"Feature set {set(currentSet)} was best, accuracy is {bestAccuracy * 100:.1f}%\n")

    print(f"Finished search! The best feature subset is {set(bestOverallSubset)}, which has an accuracy of {bestOverallAccuracy * 100:.1f}%")
    return currentFeaturesAndAccuracy


def backwardEliminationSearch(df):
    dfNumRows = df.shape[0]
    dfNumCols = df.shape[1] - 1

    # Start with all features; one is removed per level
    features = list(range(1, dfNumCols + 1))
    # Ordered list of (feature, accuracy) as features are greedily removed
    currentFeaturesAndAccuracy = []

    bestOverallAccuracy = 0.0
    bestOverallSubset = []

    print(f"\nThis dataset has {dfNumCols} features (not including the class attribute), with {dfNumRows} instances.")

    # Accuracy with all features upfront
    allFeaturesAccuracy = leaveOneOutCrossValidation(df, list(range(1, dfNumCols)), dfNumCols)
    print(f'\nRunning nearest neighbor with all {dfNumCols} features, using "leaving-one-out" evaluation, I get an accuracy of {allFeaturesAccuracy * 100:.1f}%')

    print("\nBeginning Backward Elimination search.\n")

    # Outer loop: each level greedily removes the single worst remaining feature
    for level in range(1, dfNumCols + 1):
        featureIndex = 0
        bestIndex = 0
        bestAccuracy = 0.0

        # Inner loop: evaluate each feature as the candidate to remove
        for feature in features:
            # candidateSet is all remaining features except the one being tested for removal
            candidateSet = [f for f in features if f != feature]

            # Edge case: removing the last feature leaves an empty set — skip evaluation
            if len(candidateSet) == 0:
                bestIndex = featureIndex
                bestAccuracy = 0.0
                featureIndex += 1
                continue

            accuracy = leaveOneOutCrossValidation(df, candidateSet[:-1], candidateSet[-1])
            print(f"\tUsing feature(s) {set(candidateSet)} accuracy is {accuracy * 100:.1f}%")

            # Track which removal produced the highest accuracy at this level
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                bestIndex = featureIndex
            featureIndex += 1

        # Commit the removal: pop the worst feature out of the active set
        item = features.pop(bestIndex)
        currentFeaturesAndAccuracy.append((item, bestAccuracy))

        # Warn if accuracy dropped, but continue searching in case of local maxima
        if bestAccuracy < bestOverallAccuracy:
            print(f"\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        else:
            bestOverallAccuracy = bestAccuracy
            # bestOverallSubset is the remaining features after the removal
            bestOverallSubset = features[:]

        print(f"Feature set {set(features)} was best, accuracy is {bestAccuracy * 100:.1f}%\n")

    print(f"Finished search! The best feature subset is {set(bestOverallSubset)}, which has an accuracy of {bestOverallAccuracy * 100:.1f}%")
    return currentFeaturesAndAccuracy

def main():
    df = readDatasetAndCreateDataframe()
    featuresAndAccuracy = featureSelectionSearch(df)

    print(featuresAndAccuracy)

if __name__ == "__main__":
    main()
