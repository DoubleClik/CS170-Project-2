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

def forwardSelectionSearch(df):
    dfNumRows = df.shape[0]
    dfNumCols = df.shape[1] - 1 # Exclude class label column 0
    
    features = list(range(1, dfNumCols + 1)) # List of unchosen features
    currentFeaturesAndAccuracy = [] # List of (chosen feature, bestAccuracy) tuples for chosen features

    for level in range(1, dfNumCols + 1):
        # print(f"On level {level} of the search tree")
        featureIndex = 0
        bestIndex = 0
        bestAccuracy = 0.0

        for feature in features:
            # print(f"--Considering adding feature {feature}")
            accuracy = leaveOneOutCrossValidation(df, [feature for feature, _ in currentFeaturesAndAccuracy], feature)
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                bestIndex = featureIndex
            featureIndex += 1

        item = features.pop(bestIndex)
        currentFeaturesAndAccuracy.append((item, bestAccuracy))
        # print(f"On level {level} added feature {item} to current set")

    return currentFeaturesAndAccuracy

def backwardEliminationSearch(df):
    dfNumRows = df.shape[0]
    dfNumCols = df.shape[1] - 1  # Exclude class label column 0

    features = list(range(1, dfNumCols + 1))  # Start with ALL features
    currentFeaturesAndAccuracy = []            # Tracks (removed feature, bestAccuracy)

    for level in range(1, dfNumCols + 1):
        featureIndex = 0
        bestIndex = 0
        bestAccuracy = 0.0

        for feature in features:
            candidateSet = [f for f in features if f != feature]
            
            # Skip if no features left to evaluate
            if len(candidateSet) == 0:
                bestIndex = featureIndex
                bestAccuracy = 0.0
                featureIndex += 1
                continue

            accuracy = leaveOneOutCrossValidation(df, candidateSet[:-1], candidateSet[-1])
            
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                bestIndex = featureIndex
            featureIndex += 1

        item = features.pop(bestIndex)
        currentFeaturesAndAccuracy.append((item, bestAccuracy))

    return currentFeaturesAndAccuracy

def main():
    df = readDatasetAndCreateDataframe()
    featuresAndAccuracy = backwardEliminationSearch(df)

    print(featuresAndAccuracy)

if __name__ == "__main__":
    main()
