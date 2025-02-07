import random

import numpy as np
from scipy import stats


class DataQualityAnalyzer:

    @staticmethod
    def analyze_missing_values(df):
        # perform check using self.config if needed
        missing_values = df.isnull().sum()

        if missing_values.any():
            print('Data has missing values')
        else:
            print('Data has no missing values ✅')

        return df


    @staticmethod
    def analyze_id_duplicates(df, fix_issue=False):

        duplicate_ids_indexes = df[df.duplicated(subset=['ID'], keep=False)].index

        if duplicate_ids_indexes.any():
            print('Duplicated IDs were found')

            if fix_issue:
                print('Fixing duplicate IDs')
                existing_ids = set(df["ID"])

                def generate_unique_id():
                    while True:
                        new_id = random.randint(1000, 9999)
                        if new_id not in existing_ids:
                            existing_ids.add(new_id)

                            return new_id

                for idx in duplicate_ids_indexes:
                    df.at[idx, "ID"] = generate_unique_id()

                print('Duplicated IDs were fixed.')
        else:
            print('Data has no duplicate IDs ✅')

        return df


    @staticmethod
    def analyze_duplicate_rows(df, fix_issue=False):
        duplicate_rows = df.duplicated().sum()

        if duplicate_rows > 0:
            print('Data has duplicated rows', duplicate_rows)
            if fix_issue:
                df = df.drop_duplicates().reset_index(drop=True)
                print("Duplicate rows have been removed.")
        else:
            print('Data has no duplicated rows ✅')

        return df


    @staticmethod
    def analyze_logical_issues(df, fix_issue=False):
        print(df.columns, 'Marital_Status_YOLO' in df.columns)
        negative_mask = df < 0

        negative_count_per_row = negative_mask.sum(axis=1)

        # Get the row indices with negative values
        rows_with_negatives = negative_count_per_row[negative_count_per_row > 0]

        # Show the number of negative values per row and the corresponding row indices
        if rows_with_negatives.any():
            print("Rows with negative values and their counts:")
            print(rows_with_negatives)

            if fix_issue:
                print('Deleting rows with negative values')
                df = df[~negative_mask.any(axis=1)]
                print('Rows with negative values were deleted.')

        if 'Marital_Status_YOLO' in df.columns or 'Marital_Status_Absurd' in df.columns:
            mask_yolo_absurd = (df['Marital_Status_YOLO'] == 1) | (df['Marital_Status_Absurd'] == 1)
            rows_with_yolo_absurd = df[mask_yolo_absurd]

            if not rows_with_yolo_absurd.empty:
                print("Data has invalid marital statuses")

                if fix_issue:
                    print('Removing rows with invalid marital statuses')
                    df = df[~mask_yolo_absurd]
        else:
            print('Data has no logical issues ✅')

        return df


    @staticmethod
    def analyze_outliers(df, fix_issue=False):
        outlier_indices = {}  # Dictionary to store outlier indices for each column

        for column in df.columns:  # Only numerical columns
            outlier_indices[column] = []  # Initialize the list for storing outlier indices

            # Skip columns where the standard deviation is zero or very close to zero (precision loss)
            if np.isclose(df[column].std(), 0, atol=1):  # Adjust atol as needed
                # print(f"Skipping column {column} due to near-zero standard deviation.")
                continue

            z_scores = stats.zscore(df[column])
            outliers = df[(z_scores > 5) | (z_scores < -5)]
            outlier_indices[column] = outliers.index.tolist()

            if outlier_indices[column] and fix_issue:
                print('Removing rows with outliers:', column, outlier_indices[column])
                df = df.drop(outlier_indices[column])

        return df


