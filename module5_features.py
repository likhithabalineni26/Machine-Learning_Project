def create_features(df):
    print("\n[MODULE 5] FEATURE ENGINEERING")

    df = df.copy()

    # Useful derived feature
    df["Fertilizer_per_Area"] = df["Fertilizer"] / df["Area"].replace(0, 1)

    # Interaction between two weather variables
    df["Rainfall_Temperature"] = df["Rainfall"] * df["Temperature"]

    print("Created features:")
    print("- Fertilizer_per_Area")
    print("- Rainfall_Temperature")

    return df
