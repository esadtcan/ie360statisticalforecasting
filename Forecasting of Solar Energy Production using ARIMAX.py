import pandas as pd
import numpy as np
from datetime import datetime
from forecast_helper import (
    get_production_data,
    get_historical_weather,
    get_weather_forecast,
)
import statsmodels.api as sm


# Adjust the forecasting day.
### date_to_be_forecasted defnes the day that your model should predict.
### Testing whether your model works well for the past days might be useful.

date_to_be_forecasted = str(datetime.now() + pd.DateOffset(1))[:10]
# date_to_be_forecasted = "2025-03-12"
last_datetime_to_be_forecasted = pd.to_datetime(date_to_be_forecasted) + pd.DateOffset(
    days=1, hours=-1
)

current_date = datetime.now().date()
print("Current date:", datetime.now())
print("Last date and hour to be forecasted:", last_datetime_to_be_forecasted)

# Get production data (dependent variable)
production_data = get_production_data()


# Get weather data
### Here we set get_forecast_data True. It means that we use forecast history as past data.
### You can set it False if you want to use actual historical weather data.
start_date = "2022-01-01"
plant_coordinates = [
    [37.76967344769773, 33.574324027291595],
    [37.797667446516925, 34.46742393001458],
]
weather_forecast_models = ["ecmwf_ifs025"]
weather_variables = [
    "temperature_2m",
    "shortwave_radiation",
    "cloudcover",
    "relativehumidity_2m",
    "weathercode",
]


meteo_data_historical = get_historical_weather(
    start_date=start_date,
    variables=weather_variables,
    coordinates=plant_coordinates,
    get_forecast_data=True,
)

meteo_data_future = get_weather_forecast(
    forecast_days=6,
    past_days=30,
    variables=weather_variables,
    coordinates=plant_coordinates,
    models=weather_forecast_models,
)


meteo_data_historical = meteo_data_historical.dropna()
meteo_data_future = meteo_data_future.dropna()

### Here we are combining past and future tables
### For any given date, we use the historical data if it is available, otherwise we will use the past days of forecast
meteo_data_historical.insert(0, "type", "historical")
meteo_data_future.insert(0, "type", "future")
meteo_data_all = pd.concat([meteo_data_historical, meteo_data_future], axis=0)
meteo_data_all["priorty"] = meteo_data_all.groupby("dt")["type"].rank(ascending=False)
meteo_data_all = meteo_data_all[meteo_data_all["priorty"] == 1]
meteo_data_all = meteo_data_all.sort_values("dt")
meteo_data_all = meteo_data_all.drop(["type", "priorty"], axis=1)

# Prepare the main data table
### First date in the table is the same with meteorology data.
### The last date in the main table is the day we will predict (d+1)


df_dates = pd.date_range(start_date, last_datetime_to_be_forecasted, freq="1h")
df_dates = df_dates.tz_localize("Europe/Istanbul")

df = pd.DataFrame()
df["dt"] = df_dates

### Adding the production data to main table
df = df.merge(production_data, how="left")

### Adding the meteorology data
df = df.merge(meteo_data_all, how="left")

### We use the last available data point in the data (3 days ago)
df["sun_rt_lag_3days"] = df["sun_rt"].shift(3 * 24)

# df.head(20)

# TEMPERATURE: Ortalaması ve maksimumu
df["temperature_2m_mean"] = df[
    ["location_000 temperature_2m", "location_001 temperature_2m"]
].mean(axis=1)

df["temperature_2m_max"] = df[
    ["location_000 temperature_2m", "location_001 temperature_2m"]
].max(axis=1)

# SHORTWAVE RADIATION: Ortalaması
df["shortwave_radiation_mean"] = df[
    ["location_000 shortwave_radiation", "location_001 shortwave_radiation"]
].mean(axis=1)

# CLOUDCOVER: Ortalaması
df["cloudcover_mean"] = df[["location_000 cloudcover", "location_001 cloudcover"]].mean(
    axis=1
)

# RELATIVE HUMIDITY: Ortalaması
df["relativehumidity_2m_mean"] = df[
    ["location_000 relativehumidity_2m", "location_001 relativehumidity_2m"]
].mean(axis=1)

# WEATHERCODE: Ortalaması (sayısal değer olarak)
df["weathercode_mean"] = df[
    ["location_000 weathercode", "location_001 weathercode"]
].mean(axis=1)
df["effective_radiation"] = df["shortwave_radiation_mean"] * (
    1 - df["cloudcover_mean"] / 100
)

df["Date"] = df["dt"].dt.date  # Sadece tarih (YYYY-MM-DD)
df["Hour"] = df["dt"].dt.hour  # Sadece saat (0–23)

df = df.drop(
    [
        "location_000 temperature_2m",
        "location_001 temperature_2m",
        "location_000 shortwave_radiation",
        "location_001 shortwave_radiation",
        "location_000 cloudcover",
        "location_001 cloudcover",
        "location_000 relativehumidity_2m",
        "location_001 relativehumidity_2m",
        "location_000 weathercode",
        "location_001 weathercode",
    ],
    axis=1,
)

### Preparing the train data
### Here dropping rows with NA values will remove the test data in our case
### You can directly filter train data with other ways.
### It is highly recommended to check if the resulting table is as you want
train_X = df.dropna().drop(["dt", "sun_rt"], axis=1)
train_y = df.dropna()["sun_rt"]

print(f"Variables used in model: {train_X.columns}")
# train
train_X = df.dropna().drop(["dt", "sun_rt", "Date"], axis=1)
train_y = df.dropna()["sun_rt"]
train_X = sm.add_constant(train_X)


train_X = train_X.astype(float)

# test (next day)
next_day_X = df.iloc[-24:].drop(["dt", "sun_rt", "Date"], axis=1)
next_day_X = sm.add_constant(next_day_X)
next_day_X = next_day_X.astype(float)  # bu önemli!


# next_day_pred = results.predict(next_day_X)


# next_day_pred = np.maximum(next_day_pred,0)

### You should print your results as a python list as the following code
### This should be the last line of your code!
### Since my array is a numpy array, I converted it to list using .tolist()
### If you use python list, yo do not need to do that
### An example output with the correct format is as follows:
### [16.7, 21.2, 25.9, 28.3, 27.88, 28.7442, 169.0, 986.21, 1836.91, 2121.8003, 2310, 2426.3, 2459.96, 2406.9043, 2242.0599, 2127.60711, 2037.434, 1421.077, 455.7, 45.8007, 0.0, 4.2, 20.4, 21.9]
# print(next_day_pred.tolist())
# print(train_X.columns.tolist())

# Sadece sayısal sütunları seç (NaN olan satırları da çıkar)
numeric_df = df[
    [
        "sun_rt_lag_3days",
        "effective_radiation",
        "temperature_2m_mean",
        "temperature_2m_max",
        "shortwave_radiation_mean",
        "cloudcover_mean",
        "relativehumidity_2m_mean",
        "weathercode_mean",
    ]
].dropna()

# Korelasyon matrisi
corr_matrix = numeric_df.corr()

"""# Görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix")
plt.show()"""
# df

"""import matplotlib.pyplot as plt

# Tarih filtrelemesi
mask = (df["dt"].dt.date >= pd.to_datetime("2021-01-01").date()) & (df["dt"].dt.date <= pd.to_datetime("2024-04-01").date())
filtered_df = df[mask]

# Çizim
plt.figure(figsize=(14,6))
plt.plot(filtered_df["dt"], filtered_df["sun_rt"], label="Sun Radiation")
plt.title("Güneş Üretimi (2023-07-01 - 2024-07-01)")
plt.xlabel("Tarih")
plt.ylabel("Üretim (sun_rt)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()"""
df["Date"] = df["dt"].dt.date  # ya da: pd.to_datetime(df["dt"]).dt.date
# print(df[["Date", "temperature_2m_mean"]].dropna().head())

df_hourly = df
# df_hourly
# Date sütununu oluştur
df["Date"] = df["dt"].dt.date

# Gruplama ve özet tablo
daily_series = (
    df.groupby("Date")
    .agg(
        total_sun_rt=("sun_rt", "sum"),
        max_radiation=("shortwave_radiation_mean", "max"),
        sun_rt_lag_3days_sum=("sun_rt_lag_3days", "sum"),
        temperature_2m_mean=("temperature_2m_mean", "mean"),
        shortwave_radiation_mean=("shortwave_radiation_mean", "mean"),
        cloudcover_mean=("cloudcover_mean", "mean"),
        relativehumidity_2m_mean=("relativehumidity_2m_mean", "mean"),
        weathercode_mean=("weathercode_mean", "mean"),
        effective_radiation=("effective_radiation", "mean"),
    )
    .reset_index()
)

# Sonuçları göster
# daily_series.tail()
"""import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(daily_series["Date"], daily_series["total_sun_rt"], color='orange', linewidth=2)
plt.title("Günlük Toplam Güneş Üretimi")
plt.xlabel("Tarih")
plt.ylabel("Toplam Güneş Üretimi (sun_rt)")
plt.grid(True)
plt.tight_layout()
plt.show()"""
"""import seaborn as sns
import matplotlib.pyplot as plt

# Sayısal sütunlar üzerinden otomatik seçim yapabiliriz
numeric_cols = daily_series.select_dtypes(include=["float64", "float32", "int"]).columns

# Scatterplot matrix (ggpairs alternatifi)
sns.pairplot(daily_series[numeric_cols], diag_kind='kde')
plt.suptitle("Günlük Veriler Arası İlişki Matrisi", y=1.02)
plt.show()"""

"""import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.regplot(data=daily_series, x="cloudcover_mean", y="total_sun_rt", scatter_kws={"alpha":0.6})
plt.title("Radyasyon vs Güneş Üretimi")
plt.xlabel("Kısa Dalga Radyasyon Ortalaması")
plt.ylabel("Günlük Toplam Güneş Üretimi")
plt.grid(True)
plt.tight_layout()
plt.show()"""
"""import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.regplot(data=daily_series, 
            x="max_radiation", 
            y="total_sun_rt", 
            lowess=True,          # LOESS eğrisi
            scatter_kws={"alpha":0.6},
            line_kws={"color": "red"})
plt.title("Maksimum Radyasyon vs Günlük Güneş Üretimi (LOESS)")
plt.xlabel("Maksimum Radyasyon")
plt.ylabel("Toplam Güneş Üretimi")
plt.grid(True)
plt.tight_layout()
plt.show()"""
# 1. Trend sütunu (1'den başlayarak sıra numarası)
daily_series["trnd"] = range(1, len(daily_series) + 1)

# 2. Haftanın günü (kısa ad)
daily_series["w_day"] = daily_series["Date"].apply(
    lambda x: x.strftime("%a")
)  # örn: Mon, Tue

# 3. Ay adı (kısa ad)
daily_series["mon"] = daily_series["Date"].apply(
    lambda x: x.strftime("%b")
)  # örn: Jan, Feb

# Sonuçları göster
# daily_series.sample()
# Bağımsız ve bağımlı değişkenleri ayır
X = daily_series["trnd"]
y = daily_series["total_sun_rt"]

# Sabit terimi (intercept) ekle
X = sm.add_constant(X)

# Modeli oluştur ve eğit
lm_base = sm.OLS(y, X).fit()

# Özet tablo
# print(lm_base.summary())
tmp = daily_series.copy()

daily_series["log_total_sun_rt"] = np.log1p(daily_series["total_sun_rt"])  # log(1 + y)
daily_series["log_sun_rt_lag_3days_sum"] = np.log1p(
    daily_series["sun_rt_lag_3days_sum"]
)  # log(1 + y)

# Log-transform bağımsız değişkenler
daily_series["log_total_sun_rt"] = np.log1p(daily_series["total_sun_rt"])
daily_series["log_max_radiation"] = np.log1p(daily_series["max_radiation"])
daily_series["log_sun_rt_lag_3days_sum"] = np.log1p(
    daily_series["sun_rt_lag_3days_sum"]
)
daily_series["log_temperature_2m_mean"] = np.log1p(daily_series["temperature_2m_mean"])
daily_series["log_shortwave_radiation_mean"] = np.log1p(
    daily_series["shortwave_radiation_mean"]
)
daily_series["log_cloudcover_mean"] = np.log1p(daily_series["cloudcover_mean"])
daily_series["log_relativehumidity_2m_mean"] = np.log1p(
    daily_series["relativehumidity_2m_mean"]
)
daily_series["log_weathercode_mean"] = np.log1p(daily_series["weathercode_mean"])
daily_series["log_effective_radiation"] = np.log1p(daily_series["effective_radiation"])

tmp["log_total_sun_rt"] = np.log1p(tmp["total_sun_rt"])
tmp["log_sun_rt_lag_3days_sum"] = np.log1p(tmp["sun_rt_lag_3days_sum"])

# Bağımlı ve bağımsız değişkenlerin log dönüşümü
tmp["log_total_sun_rt"] = np.log1p(tmp["total_sun_rt"])
tmp["log_max_radiation"] = np.log1p(tmp["max_radiation"])
tmp["log_sun_rt_lag_3days_sum"] = np.log1p(tmp["sun_rt_lag_3days_sum"])
tmp["log_temperature_2m_mean"] = np.log1p(tmp["temperature_2m_mean"])
tmp["log_shortwave_radiation_mean"] = np.log1p(tmp["shortwave_radiation_mean"])
tmp["log_cloudcover_mean"] = np.log1p(tmp["cloudcover_mean"])
tmp["log_relativehumidity_2m_mean"] = np.log1p(tmp["relativehumidity_2m_mean"])
tmp["log_weathercode_mean"] = np.log1p(tmp["weathercode_mean"])
tmp["log_effective_radiation"] = np.log1p(tmp["effective_radiation"])
import statsmodels.api as sm
import numpy as np
import pandas as pd

# 1. y_log oluştur
y_log = daily_series["log_total_sun_rt"]

# 2. X_log oluştur (kategorik + sürekli değişkenler)
w_day_dummies = pd.get_dummies(daily_series["w_day"], prefix="w_day", drop_first=False)

X_log = pd.concat(
    [
        daily_series[
            [
                "trnd",
                "log_max_radiation",
                "log_sun_rt_lag_3days_sum",
                "log_temperature_2m_mean",
                "log_shortwave_radiation_mean",
                "log_cloudcover_mean",
                "log_relativehumidity_2m_mean",
                "log_weathercode_mean",
                "log_effective_radiation",
            ]
        ],
        w_day_dummies,
    ],
    axis=1,
)
X_log.replace([np.inf, -np.inf], np.nan, inplace=True)
y_log.replace([np.inf, -np.inf], np.nan, inplace=True)

# Ortak NaN'leri düşür
valid_mask = X_log.notna().all(axis=1) & y_log.notna()
X_log = X_log[valid_mask]
y_log = y_log[valid_mask]

X_log = X_log.astype(float)
y_log = y_log.astype(float)

lm_log = sm.OLS(y_log, X_log).fit()


# ----------- tmp için tahmin ----------

# Aynı işlemleri tmp için yap
w_day_dummies_tmp = pd.get_dummies(tmp["w_day"], prefix="w_day", drop_first=False)

X_tmp_log = pd.concat(
    [
        tmp[
            [
                "trnd",
                "log_max_radiation",
                "log_sun_rt_lag_3days_sum",
                "log_temperature_2m_mean",
                "log_shortwave_radiation_mean",
                "log_cloudcover_mean",
                "log_relativehumidity_2m_mean",
                "log_weathercode_mean",
                "log_effective_radiation",
            ]
        ],
        w_day_dummies_tmp,
    ],
    axis=1,
)

# Kolonlar otomatik hizalanıyordu, burada biz yapıyoruz:
X_tmp_log = X_tmp_log.reindex(columns=X_log.columns, fill_value=0)

# Tahmin
tmp["log_predicted"] = lm_log.predict(X_tmp_log)
# np.expm1() çalışabilmesi için dtype'ı float yap
tmp["log_predicted"] = tmp["log_predicted"].astype(float)

# Ardından dönüşüm:
tmp["predicted_from_log"] = np.expm1(tmp["log_predicted"])
tmp["residual_from_log"] = tmp["total_sun_rt"] - tmp["predicted_from_log"]

"""import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

# Gerçek değerler
plt.plot(tmp["Date"], tmp["total_sun_rt"], label="Gerçek", color="black", linewidth=2)

# Tahmin değerleri
plt.plot(tmp["Date"], tmp["predicted_trend_day"], label="Tahmin (Regresyon)", color="orange", linestyle="--", linewidth=2)

plt.title("Günlük Güneş Üretimi: Gerçek vs. Model Tahmini (Trend + Gün + Meteoroloji)")
plt.xlabel("Tarih")
plt.ylabel("Güneş Üretimi (sun_rt)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
wmape_log = (tmp["residual_from_log"].abs().sum() / tmp["total_sun_rt"].sum()) * 100
print(f"WMAPE (Log Modelli): {wmape_log:.2f}%")"""
# print(tmp)
residuals_log = tmp["residual_from_log"].dropna()
"""from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# ACF
plt.figure(figsize=(12, 4))
plot_acf(residuals_log, lags=40)
plt.title("Kalıntıların ACF Grafiği (Log Modelli)")
plt.tight_layout()
plt.show()

# PACF
plt.figure(figsize=(12, 4))
plot_pacf(residuals_log, lags=40, method="ywm")
plt.title("Kalıntıların PACF Grafiği (Log Modelli)")
plt.tight_layout()
plt.show()"""
from statsmodels.tsa.stattools import kpss

# Kalıntıları al
residuals_log = tmp["residual_from_log"].dropna()

# KPSS testi (trend varsayılan olarak False)
stat, p_value, lags, crit = kpss(residuals_log, regression="c", nlags="auto")

# print("KPSS Testi Sonuçları:")
# print(f"Test İstatistiği: {stat:.4f}")
# print(f"P-Değeri: {p_value:.4f}")
# print(f"Kritik Değerler: {crit}")
from statsmodels.stats.diagnostic import acorr_ljungbox

ljung = acorr_ljungbox(residuals_log, lags=[10, 20, 30], return_df=True)

# print("\nLjung-Box Testi Sonuçları:")
# print(ljung)

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Gerekli: NaN veya inf varsa temizle
X_log_2 = X_log.copy()
X_log.replace([np.inf, -np.inf], np.nan, inplace=True)
X_log.dropna(inplace=True)

# Bağımlı değişkeni hizala
y_log_aligned = y_log.loc[X_log.index].squeeze()  # squeeze: Series'e çevirir

# Y değeri sıfır olan (tahmin yapılacak) günleri çıkar
# Gerçek (0 olmayan) günleri filtrele
mask = y_log > 0

X_train = X_log.loc[mask]
y_train = y_log.loc[mask].squeeze()


from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    y_train,
    exog=X_train,
    order=(1, 1, 1),
    enforce_stationarity=False,
    enforce_invertibility=False,
)

results = model.fit(disp=False)

y_pred_log = results.fittedvalues
y_pred = np.expm1(y_pred_log)  # Orijinal ölçekte tahminler

# Gerçek ve tahmin değerlerinin farkı (örneğin eğitim seti için)
tmp.loc[y_train.index, "resid_arimax"] = np.expm1(y_train) - y_pred

# print(tmp)

"""import matplotlib.pyplot as plt

# Gerçek ve tahmin değerlerini al (NaN olmayanları)
mask = tmp["resid_arimax"].notna() & (tmp["total_sun_rt"] > 0)
actual_values = tmp.loc[mask, "total_sun_rt"]
predicted_values = tmp.loc[mask, "predicted_from_log"]
dates = tmp.loc[mask, "Date"]

# Plot
plt.figure(figsize=(15, 6))
plt.plot(dates, actual_values, label="Gerçek Değer", linewidth=2)
plt.plot(dates, predicted_values, label="Tahmin (ARIMAX)", linestyle="--", linewidth=2)
plt.xlabel("Tarih")
plt.ylabel("Güneş Enerjisi Üretimi")
plt.title("Gerçek vs ARIMAX Tahmin")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

wmape_arimax = (tmp["resid_arimax"].abs().sum() / tmp["total_sun_rt"].sum()) * 100
print(f"WMAPE (ARIMAX Modeli): {wmape_arimax:.2f}%")"""
# print(daily_series)
# print(tmp)
# y_train
y_extended = y_train.copy()  # Sadece gerçek veriler
future_indices = tmp[tmp["total_sun_rt"] == 0].index  # 03.05 ve 04.05
for idx in future_indices:
    X_i = X_log.loc[[idx]]

    model_i = SARIMAX(
        y_extended,
        exog=X_log.loc[y_extended.index],
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    forecast_log = model_i.forecast(steps=1, exog=X_i)
    y_extended.loc[idx] = forecast_log.values[0]  # log ölçeğinde ekle
# 1. Tahminleri tmp'ye ekle (log scale)
tmp.loc[future_indices, "log_total_sun_rt_predicted"] = y_extended.loc[future_indices]

# 2. Orijinal skala: expm1 ile
tmp["predicted_recursive"] = np.expm1(tmp["log_total_sun_rt_predicted"])

# 3. Gerçek ve tahmin karşılaştırması (sadece tahmin edilen günler)
# print(tmp.loc[future_indices, ["Date", "predicted_recursive"]])
"""plt.figure(figsize=(15, 6))

# Gerçek geçmiş değerler
mask = tmp["resid_arimax"].notna()
plt.plot(tmp.loc[mask, "Date"], tmp.loc[mask, "total_sun_rt"], label="Gerçek Değer", linewidth=2)

# ARIMAX tahminleri (geçmiş için)
plt.plot(tmp.loc[mask, "Date"], tmp.loc[mask, "predicted_from_log"], label="Tahmin (ARIMAX)", linestyle="--", linewidth=2)

# Yeni tahminler (gelecek)
plt.scatter(tmp.loc[future_indices, "Date"], tmp.loc[future_indices, "predicted_recursive"], color="red", label="Gelecek Tahmin", zorder=5)

plt.xlabel("Tarih")
plt.ylabel("Güneş Enerjisi Üretimi")
plt.title("Geçmiş ve Gelecek Tahminler")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
# Gerçek ve tahmin değerlerinin olduğu satırları filtrele
mask_valid = (tmp["total_sun_rt"] > 0) & tmp["predicted_from_log"].notna()

# WMAPE hesapla
wmape_total = (
    (tmp.loc[mask_valid, "total_sun_rt"] - tmp.loc[mask_valid, "predicted_from_log"]).abs().sum()
    / tmp.loc[mask_valid, "total_sun_rt"].sum()
) * 100

print(f"Toplam WMAPE (ARIMAX Modeli): {wmape_total:.2f}%")"""
# print(tmp)
tmp["Date"] = pd.to_datetime(tmp["Date"])
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

# 1. Özellik oluşturma
features = [
    "log_max_radiation",
    "log_sun_rt_lag_3days_sum",
    "log_temperature_2m_mean",
    "log_shortwave_radiation_mean",
    "log_cloudcover_mean",
    "log_relativehumidity_2m_mean",
    "log_weathercode_mean",
    "log_effective_radiation",
]

tmp["trnd"] = range(1, len(tmp) + 1)
tmp["w_day"] = pd.to_datetime(tmp["Date"]).dt.day_name()
w_day_dummies = pd.get_dummies(tmp["w_day"], prefix="C(w_day)")

# X (bağımsız değişkenler) hazırla
X_all = pd.concat([w_day_dummies, tmp[features], tmp[["trnd"]]], axis=1).astype(float)
X_all = X_all.replace([np.inf, -np.inf], np.nan)

# Tarihleri datetime'a çevir
tmp["Date"] = pd.to_datetime(tmp["Date"])

# Son 90 gün için t+1 tahmini yapılacak
test_days = tmp["Date"].iloc[-11:-1].reset_index(drop=True)  # t günleri
forecast_results = []

for t_day in test_days:
    t_plus_1_day = t_day + timedelta(days=1)
    t_minus_2_day = t_day - timedelta(days=2)

    # Eğitim verisi: sadece t-2 ve öncesi
    train_mask = tmp["Date"] <= t_minus_2_day
    valid_train = (
        train_mask & (~X_all.isna().any(axis=1)) & (~tmp["total_sun_rt"].isna())
    )

    if valid_train.sum() < 30:
        forecast_results.append(
            {"forecast_date": t_plus_1_day, "forecast": np.nan, "actual": np.nan}
        )
        continue

    y_train = np.log1p(tmp.loc[valid_train, "total_sun_rt"]).astype(float)
    X_train = X_all.loc[valid_train].astype(float)

    # Tahmin yapılacak günün exog verisi
    future_idx_list = tmp.index[tmp["Date"] == t_plus_1_day].tolist()
    if not future_idx_list:
        forecast_results.append(
            {"forecast_date": t_plus_1_day, "forecast": np.nan, "actual": np.nan}
        )
        continue

    future_idx = future_idx_list[0]
    if future_idx not in X_all.index or pd.isna(X_all.loc[future_idx]).any():
        forecast_results.append(
            {"forecast_date": t_plus_1_day, "forecast": np.nan, "actual": np.nan}
        )
        continue

    X_i = X_all.loc[[future_idx]]
    y_i = tmp.loc[future_idx, "total_sun_rt"]
    y_i_log = np.log1p(y_i) if not pd.isna(y_i) else np.nan

    try:
        model = SARIMAX(
            y_train,
            exog=X_train,
            order=(1, 1, 1),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        y_hat_log = model.forecast(steps=1, exog=X_i)
        y_hat = np.expm1(y_hat_log.values[0])
        forecast_results.append(
            {"forecast_date": t_plus_1_day, "forecast": y_hat, "actual": y_i}
        )
    except:
        forecast_results.append(
            {"forecast_date": t_plus_1_day, "forecast": np.nan, "actual": y_i}
        )

# Sonuçları DataFrame olarak topla
forecast_df = pd.DataFrame(forecast_results)
# print(forecast_df.tail())


"""import matplotlib.pyplot as plt

# Sıfır olmayan gerçek değerlerle çalış
filtered_df = forecast_df[(forecast_df["actual"] > 0) & (~forecast_df["forecast"].isna())]

plt.figure(figsize=(12, 6))
plt.plot(filtered_df["forecast_date"], filtered_df["forecast"], label="Forecast", marker='o')
plt.plot(filtered_df["forecast_date"], filtered_df["actual"], label="Actual", marker='x')

plt.title("ARIMAX Forecast vs Actual (t+1 prediction using only t-2 actual)")
plt.xlabel("Forecast Date")
plt.ylabel("Total Sun Radiation")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# WMAPE hesapla (actual değeri sıfır olanları dışla)
valid_mask = (
    ~forecast_df["actual"].isna() &
    ~forecast_df["forecast"].isna() &
    (forecast_df["actual"] > 0)
)

wmape = (
    np.sum(np.abs(forecast_df.loc[valid_mask, "actual"] - forecast_df.loc[valid_mask, "forecast"])) /
    np.sum(forecast_df.loc[valid_mask, "actual"])
) * 100

print(f"WMAPE (Sadece t-2 güne kadar actual sun_rt, t+1 tahmini, Aggregated 90 gün): {wmape:.2f}%")"""
# df
# 1. Saatlik oran profillerini oluştur
df["Date"] = pd.to_datetime(df["Date"])
df["Hour"] = df["dt"].dt.hour
df["week"] = df["Date"].dt.isocalendar().week
df["year"] = df["Date"].dt.isocalendar().year

# Önceden varsa çakışmayı önlemek için sil
if "daily_total" in df.columns:
    df = df.drop(columns=["daily_total"])

# Günlük toplam üretimi hesapla ve merge et
daily_totals = df.groupby("Date")["sun_rt"].sum().rename("daily_total")
df = df.merge(daily_totals, on="Date")

# Saatlik oran = saatlik üretim / günlük toplam
df["hourly_ratio"] = df["sun_rt"] / df["daily_total"]

# 2. year + week + hour bazlı saatlik oran ortalamasını al
weekly_hourly_profile = (
    df.groupby(["year", "week", "Hour"])["hourly_ratio"].mean().reset_index()
)

# Kontrol için ilk 5 satırı yazdır
# print(weekly_hourly_profile.head())
from datetime import timedelta

# 1. Saatlik oran profillerini hazırla (year dikkate alınmadan)
# 1 üst cellde hazır

# 2. Günlük forecast_df'i oluşturduğun ARIMA kısmını (daha önceki kodunu) kullan
# yukarıda çalıştırıldı

# Sonuçları DataFrame olarak topla
forecast_df = pd.DataFrame(forecast_results)


def distribute_to_hours_fallback(forecast_df, weekly_hourly_profile):
    forecast_df = forecast_df.copy()
    forecast_df["forecast_date"] = pd.to_datetime(forecast_df["forecast_date"])
    result_rows = []

    for _, row in forecast_df.iterrows():
        date = row["forecast_date"]
        total_forecast = row["forecast"]

        if pd.isna(total_forecast):
            continue

        iso = date.isocalendar()
        year = iso.year
        week = iso.week

        profile = weekly_hourly_profile[
            (weekly_hourly_profile["year"] == year)
            & (weekly_hourly_profile["week"] == week)
        ]

        # Fallback: 7 gün öncesi
        if profile.empty or profile["hourly_ratio"].dropna().empty:
            fallback_date = date - pd.Timedelta(days=7)
            fb_iso = fallback_date.isocalendar()
            profile = weekly_hourly_profile[
                (weekly_hourly_profile["year"] == fb_iso.year)
                & (weekly_hourly_profile["week"] == fb_iso.week)
            ]

        profile = profile.dropna(subset=["hourly_ratio"])

        if profile.empty:
            # Tüm fallback'ler de boşsa: eşit dağıt
            for h in range(24):
                result_rows.append(
                    {
                        "datetime": date + pd.Timedelta(hours=h),
                        "hourly_forecast": total_forecast / 24,
                    }
                )
        else:
            # Normalize oranlar
            ratio_sum = profile["hourly_ratio"].sum()
            profile["normalized_ratio"] = profile["hourly_ratio"] / ratio_sum

            for _, ratio_row in profile.iterrows():
                h = ratio_row["Hour"]
                ratio = ratio_row["normalized_ratio"]
                result_rows.append(
                    {
                        "datetime": date + pd.Timedelta(hours=int(h)),
                        "hourly_forecast": total_forecast * ratio,
                    }
                )

    return pd.DataFrame(result_rows)


hourly_forecast_df = distribute_to_hours_fallback(forecast_df, weekly_hourly_profile)
# 5. İsteğe bağlı: actual ile karşılaştırmak istersen
df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize(None)
actual_hourly = df.set_index("dt")[["sun_rt"]].rename(columns={"sun_rt": "actual"})

hourly_forecast_df["datetime"] = pd.to_datetime(
    hourly_forecast_df["datetime"]
).dt.tz_localize(None)
result = hourly_forecast_df.set_index("datetime").join(actual_hourly, how="left")

# 5. İsteğe bağlı: actual ile karşılaştırmak istersen
actual_hourly = df.set_index("dt")[["sun_rt"]].rename(columns={"sun_rt": "actual"})
result = hourly_forecast_df.set_index("datetime").join(actual_hourly, how="left")

# 6. WMAPE hesapla (son 72 saat hariç)
valid_mask = ~result["actual"].isna() & ~result["hourly_forecast"].isna()

# Son 72 saatlik zaman aralığını belirle
latest_times_to_exclude = result.index.sort_values()[-72:]

# Bu 72 saati valid mask'ten çıkar
valid_mask = valid_mask & (~result.index.isin(latest_times_to_exclude))
# print(actual_hourly)
# print(hourly_forecast_df)

"""import matplotlib.pyplot as plt

# Son 30 günü filtrele
plot_data = result.dropna().copy()
plot_data = plot_data.sort_index()
plot_data = plot_data.loc[
    plot_data.index >= plot_data.index.max() - pd.Timedelta(days=30)
]
# Çizim
plt.figure(figsize=(14, 6))
plt.plot(plot_data.index, plot_data["hourly_forecast"], label="Forecast", linestyle="-")
plt.plot(
    plot_data.index, plot_data["actual"], label="Actual", linestyle="--", alpha=0.8
)
plt.xlabel("Datetime")
plt.ylabel("Solar Radiation (sun_rt)")
plt.title("Hourly Forecast vs Actual (Last 90 Days)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# WMAPE hesapla
wmape_hourly = (
    np.sum(
        np.abs(
            result.loc[valid_mask, "actual"] - result.loc[valid_mask, "hourly_forecast"]
        )
    )
    / np.sum(result.loc[valid_mask, "actual"])
    * 100
)
print(f"WMAPE (Saatlik tahmin, son 72 saat hariç): {wmape_hourly:.2f}%")"""
# Son 24 saatlik tahmin değerlerini liste olarak al
last_24_forecasts = hourly_forecast_df["hourly_forecast"].iloc[-24:].round(4).tolist()
print(last_24_forecasts)
