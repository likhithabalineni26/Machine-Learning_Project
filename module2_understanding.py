def understand_dataset(df):
    print("\n[MODULE 2] DATA UNDERSTANDING")
    print("\nColumn names:")
    print(list(df.columns))

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate records:", df.duplicated().sum())

    print("\nUnique values:")
    print(df.nunique())

    print("\nStatistical summary:")
    print(df.describe(include="all").to_string())
