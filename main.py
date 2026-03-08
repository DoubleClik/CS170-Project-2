import pandas as pd

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
    dataframe = pd.read_csv(filepath, sep=r'\s+', header=None)

    return dataframe

def main():
    dataframe = readDatasetAndCreateDataframe()

    print(dataframe.head(10))

if __name__ == "__main__":
    main()
