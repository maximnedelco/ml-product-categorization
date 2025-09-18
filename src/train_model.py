import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/IMLP4_TASK_03-products.csv")
print(df.head())
print(df.info())


df.columns = df.columns.str.lower().str.strip()


print(df.info())
df.isna().sum()
df.dropna(subset= ["product title","category label"],inplace=True)



print(df["category label"].value_counts())


category_map = {
    'Fridges': 'Fridge Freezers',
    'fridge': 'Fridge Freezers',
    'Fridge Freezers': 'Fridge Freezers',
    'Mobile Phone': 'Mobile Phones',
    'Mobile Phones': 'Mobile Phones',
    'CPU': 'CPUs',
    'CPUs': 'CPUs',
    'TVs': 'TVs',
    'Freezers': 'Freezers',
    'Washing Machines': 'Washing Machines',
    'Dishwashers': 'Dishwashers',
    'Digital Cameras': 'Digital Cameras',
    'Microwaves': 'Microwaves'
}



df["category"] = df["category label"].map(category_map).fillna(df['category label'])


print(df["category"].value_counts())

print(df.dtypes)
df["category"] = df["category"].astype("category")
df["product title"] = df["product title"].astype("string")
df = df.drop(columns=["product id","merchant id", "category label","_product code","number_of_views","merchant rating",
                      "listing date"])

print(df.info())

df["title_length"] = df["product title"].astype(str).str.len()

X = df[["title_length","product title"]]
y = df["category"]


preprocessor = ColumnTransformer(
transformers=[
    ("title",TfidfVectorizer(),"product title"),
    ("lenght",MinMaxScaler(),["title_length"])
]
)

pipeline = Pipeline([
     ("preprocessing", preprocessor),
     ("classifier",RandomForestClassifier())
])


pipeline.fit(X,y)

joblib.dump(pipeline,"model/category_model.pkl",compress=3)
print("Model trained and saved as 'model/category_model.pkl'.")













