from pathlib import Path
import matplotlib.pyplot as plt

def create_eda_plots(df):
    print("\n[MODULE 4] EXPLORATORY DATA ANALYSIS")

    Path("plots").mkdir(exist_ok=True)

    # Yield distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["Yield"], bins=10)
    plt.xlabel("Crop Yield")
    plt.ylabel("Frequency")
    plt.title("Crop Yield Distribution")
    plt.tight_layout()
    plt.savefig("plots/yield_distribution.png")
    plt.close()

    # Rainfall vs Yield
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Rainfall"], df["Yield"])
    plt.xlabel("Rainfall (mm)")
    plt.ylabel("Crop Yield")
    plt.title("Rainfall vs Crop Yield")
    plt.tight_layout()
    plt.savefig("plots/rainfall_vs_yield.png")
    plt.close()

    # Crop average yield
    crop_avg = df.groupby("Crop")["Yield"].mean()
    plt.figure(figsize=(8, 5))
    crop_avg.plot(kind="bar")
    plt.xlabel("Crop")
    plt.ylabel("Average Yield")
    plt.title("Average Yield by Crop")
    plt.tight_layout()
    plt.savefig("plots/average_yield_by_crop.png")
    plt.close()

    print("EDA plots saved in plots/")
