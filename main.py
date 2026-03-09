import pandas as pd
import random

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

def leaveOneOutCrossValidation(df, currentFeatures, feature):
    return random.uniform(0, 1)

def featureSelectionSearch(df):
    dfNumRows = df.shape[0]
    dfNumCols = df.shape[1] - 1 # Exclude class label column 0
    
    features = list(range(1, dfNumCols + 1)) # List of unchosen features
    currentFeatures = [] # List of chosen features 

    for level in range(1, dfNumCols + 1):
        print(f"On level {level} of the search tree")
        featureIndex = 0
        bestIndex = 0
        bestAccuracy = 0.0

        for feature in features:
            print(f"--Considering adding feature {feature}")
            accuracy = leaveOneOutCrossValidation(0,0,0)
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                bestIndex = featureIndex
            featureIndex += 1
        
        item = features.pop(bestIndex)
        currentFeatures.append(item)
        print(f"On level {level} added feature {item} to current set")

    return currentFeatures

def main():
    df = readDatasetAndCreateDataframe()
    featuresList = featureSelectionSearch(df)
    print(featuresList)

if __name__ == "__main__":
    main()
