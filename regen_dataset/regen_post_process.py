import pandas as pd

# Đọc file CSV gốc
df = pd.read_csv("new_train.csv")

# Lọc ra những dòng không có giá trị ở cột "error"
df_success = df[df["error"].isna() | (df["error"] == "")]

# Xuất ra file mới
df_success.to_csv("train_final.csv", index=False)

print("Đã xuất", len(df_success), "dòng vào train_final.csv")