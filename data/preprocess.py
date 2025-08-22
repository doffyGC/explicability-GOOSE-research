from sklearn.preprocessing import LabelEncoder

def preprocess(df, target_column, discarted_columns):
    X = df.drop(columns=discarted_columns)
    y = df[target_column]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return X, y_encoded