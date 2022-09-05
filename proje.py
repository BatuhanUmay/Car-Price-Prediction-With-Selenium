import streamlit as st

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor


############################################################################

st.title("Car Price Prediction")

brand = st.text_input("Marka:")
series = st.text_input("Seri:")
model = st.text_input("Model:")
productionYear = st.number_input("Üretim Yılı:", format="%d", value=0)
mileage = st.number_input("Kilometre:", format="%d", value=0)
gearbox = st.selectbox("Vites tipi:", ["Düz", "Otomatik", "Yarı Otomatik"])
fuelType = st.selectbox("Yakıt türü:", ["Benzin", "Dizel", "Hibrit", "LPG & Benzin"])
bodyType = st.selectbox("Kasa tipi:", ["Coupe", "Hatchback/3", "Hatchback/5", "MPV", "Roadster", "Sedan", "Station wagon"])
engineSize = st.number_input("Motor hacmi:", format="%d", value=0)
enginePower = st.number_input("Motor gücü:", format="%d", value=0)
# enginePower = st.text_input("Motor gücü:")
drivetrain = st.selectbox("Çekiş:", ["Önden Çekiş", "Arkadan İtiş", "4WD (Sürekli)"])
fuelEfficiency = st.number_input("Ortalama yakıt tüketimi:") # kontrol et
fuelTank = st.number_input("Yakıt deposu:", format="%d", value=0) # kontrol et
replacedParts = st.text_input("Değişen parça:")
exchange = st.radio("Takasa uygunluk:", ["Takasa Uygun", "Takasa Uygun Değil"])
fromWhom = st.radio("Kimden:", ["Sahibinden", "Galeriden"])
predict = st.button("Predict!")

############################################################################

df = pd.read_csv("cars-last.csv")
df = df.iloc[:, 1:]

############################################################################

df["TL"] = df["TL"].apply(lambda x: x.replace(".", ""))
df["TL"] = df["TL"].apply(lambda x: x.replace(",", ""))
df["TL"] = df["TL"].apply(pd.to_numeric, errors="coerce")


df["Kilometre"] = df["Kilometre"].apply(lambda x: x.replace(".", ""))
df["Kilometre"] = df["Kilometre"].apply(lambda x: x.strip("km"))
df["Kilometre"] = df["Kilometre"].astype(int)


df = df[df["Motor Hacmi"].isin(["Sedan"]) == False]
df["Motor Hacmi"] = df["Motor Hacmi"].apply(lambda x: x.strip("cc"))
df["Motor Hacmi"] = df["Motor Hacmi"].astype(int)


df["Ort. Yakıt Tüketimi"] = df["Ort. Yakıt Tüketimi"].apply(lambda x: x.replace(",", "."))
df["Ort. Yakıt Tüketimi"] = df["Ort. Yakıt Tüketimi"].apply(lambda x: x.strip("lt"))
df["Ort. Yakıt Tüketimi"] = df["Ort. Yakıt Tüketimi"].astype(float)

df["Yakıt Deposu"] = df["Yakıt Deposu"].apply(lambda x: x.replace(",", "."))
df["Yakıt Deposu"] = df["Yakıt Deposu"].apply(lambda x: x.strip("lt"))
df["Yakıt Deposu"] = df["Yakıt Deposu"].astype(float)


df = df[~df['Motor Gücü'].str.contains("-")]
df = df[~df['Motor Gücü'].str.contains("ve")]
df["Motor Gücü"] = df["Motor Gücü"].apply(lambda x: x.strip("hp"))
df["Motor Gücü"] = df["Motor Gücü"].astype(int)

############################################################################

df = df.drop(["İlan No", "İlan Tarihi"], axis=1)
df = df.drop_duplicates()

df = df.dropna()
############################################################################

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

############################################################################
le = LabelEncoder()

df["Marka"] = le.fit_transform(df["Marka"])
dict_Marka = dict(zip(le.classes_, le.transform(le.classes_)))

df["Seri"] = le.fit_transform(df["Seri"])
dict_Seri = dict(zip(le.classes_, le.transform(le.classes_)))

df["Model"] = le.fit_transform(df["Model"])
dict_Model = dict(zip(le.classes_, le.transform(le.classes_)))

df["Vites Tipi"] = le.fit_transform(df["Vites Tipi"])
dict_VitesTipi = dict(zip(le.classes_, le.transform(le.classes_)))

df["Yakıt Tipi"] = le.fit_transform(df["Yakıt Tipi"])
dict_YakıtTipi = dict(zip(le.classes_, le.transform(le.classes_)))

df["Kasa Tipi"] = le.fit_transform(df["Kasa Tipi"])
dict_KasaTipi = dict(zip(le.classes_, le.transform(le.classes_)))

# df["Motor Gücü"] = le.fit_transform(df["Motor Gücü"])
# dict_MotorGucu = dict(zip(le.classes_, le.transform(le.classes_)))

df["Çekiş"] = le.fit_transform(df["Çekiş"])
dict_Cekis = dict(zip(le.classes_, le.transform(le.classes_)))

df["Boya-değişen"] = le.fit_transform(df["Boya-değişen"])
dict_BoyaDegisen = dict(zip(le.classes_, le.transform(le.classes_)))

df["Takasa Uygun"] = le.fit_transform(df["Takasa Uygun"])
dict_TakasaUygun = dict(zip(le.classes_, le.transform(le.classes_)))

df["Kimden"] = le.fit_transform(df["Kimden"])
dict_Kimden= dict(zip(le.classes_, le.transform(le.classes_)))

############################################################################

X = df.iloc[:, :-1].values # bağımsız değişkenler
y = df.iloc[:, -1:].values # bağımlı değişkenler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)

############################################################################

xgb = XGBRegressor()
xgb.fit(X_train, y_train)

############################################################################

if predict:
    print("Marka:", brand)
    print("Seri:", series)
    print("Model:", model)
    print("Üretim Yılı:", productionYear)
    print("Kilometre:", mileage)
    print("Vites tipi:", gearbox)
    print("Yakıt türü:", fuelType)
    print("Kasa tipi:", bodyType)
    print("Motor hacmi:", engineSize)
    print("Motor gücü:", enginePower)
    print("Çekiş:", drivetrain)
    print("Ortalama yakıt tüketimi:", fuelEfficiency)
    print("Yakıt deposu:", fuelTank)
    print("Değişen parça:", replacedParts)
    print("Takasa uygunluk:", exchange)
    print("Kimden:", fromWhom)

    newCar = [
        int(dict_Marka.get(brand)),
        int(dict_Seri.get(series)),
        int(dict_Model.get(model)),
        productionYear,
        mileage,
        int(dict_VitesTipi.get(gearbox)),
        int(dict_YakıtTipi.get(fuelType)),
        int(dict_KasaTipi.get(bodyType)),
        engineSize,
        # int(dict_MotorGucu.get(str(enginePower))),
        enginePower,
        int(dict_Cekis.get(drivetrain)), 
        fuelEfficiency,
        fuelTank,
        int(dict_BoyaDegisen.get(replacedParts)),
        int(dict_TakasaUygun.get(exchange)),
        int(dict_Kimden.get(fromWhom))
    ]
    newCar = np.array(newCar)
    print("New Car \n", newCar)

    y_pred = xgb.predict(newCar.reshape((1, -1)))
    print("y pred:", y_pred)

    st.write("Ürünün Fiyatı:", y_pred[0])
