import pandas as pd
import matplotlib.pyplot as plt
import os

def reduce_redundant_rows(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Get original number of rows
    original_count = len(df)

    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()

    # Get reduced number of rows
    reduced_count = len(df_cleaned)

    # Save the new file
    new_file = f"{os.path.splitext(csv_file)[0]}_new.csv"
    df_cleaned.to_csv(new_file, index=False)

    # Plot the comparison
    categories = ["Original", "Reduced"]
    values = [original_count, reduced_count]

    plt.figure(figsize=(6, 4))
    plt.bar(categories, values, color=['blue', 'green'])
    plt.ylabel("Number of Rows")
    plt.title("Row Reduction Comparison")
    plt.ylim(0, max(values) * 1.2)

    # Show values on bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.02 * max(values), str(v), ha='center', fontsize=12)

    plt.show()

    print(f"Reduced file saved as: {new_file}")

reduce_redundant_rows("./datasets/http_preprocessed.csv")