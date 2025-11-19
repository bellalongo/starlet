"""
    Script to parse through flares and select the best candidates for the training set.
"""

from os.path import exists
import pandas as pd
import re

def main():
    # Check to see if the flare directory has been created
    if not exists('flares.csv'):
        print('Creating flares.csv...')
        
        # Open flare dataset
        with open("apjac8352/datafile.txt", "r") as file:
            lines = file.readlines()

            # Iteratre through each line (except the header 0-25)
            for i, line in enumerate(lines[26:]):
                tmp = line.replace("        ", ' Nan ')
                lines[i] = tmp
        
        # Reformat lines
        cleaned_lines = [re.sub(r'\s+', ',', line.strip()) for line in lines]

        # Create header
        header = 'TIC,TESS Sector,Flare peak time (BJD),Flare amplitude (relative),Estimated flare energy 1,Estimated flare energy 2,Number of fitted flare profiles,Possible flare detection'

        # Write the cleaned data to a CSV file
        with open('flares.csv', 'w') as f:
            f.write(header + '\n')
            for line in cleaned_lines:
                f.write(line + '\n')

    # Define where the training data CSV will be stores
    training_csv = 'training.csv'

    # Check if the training data CSV exists
    if exists(training_csv):
        # Find the last line of the training data CSV and make that the starting index
        with open(training_csv, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            starting_index = int(last_line.split(',')[0])

            # Check if starting index is the last index of flares.csv
            if starting_index == len(pd.read_csv('flares.csv')):
                print(f'All {starting_index} flares have been processed!')
                return
    else:
        starting_index = 0

    # Open the flare dataset
    flare_df = pd.read_csv('flares.csv')

    # Iterate through each flare
    for _, flare in flare_df.iloc[starting_index:].iterrows():
        print(flare)
        break
            


if __name__ == '__main__':
    main()


