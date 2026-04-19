import pandas as pd

df = pd.read_csv('data/DataCoSupplyChainDataset.csv', encoding='latin-1')

# Eliminamos los valores que no aportan a este análisis
df = df.drop(columns=[
    'Order Zipcode',
    'Product Description',
    'Customer Password',
    'Product Image'
])

df['Customer Lname'] = df['Customer Lname'].fillna('Unknown')
df['Customer Zipcode'] = df['Customer Zipcode'].fillna(0)

# Nombres del apartado de shipping
print(df.columns[df.columns.str.contains('ship', case=False)].tolist())

# Crear columna de retraso
df['shipping_delay_days'] = df.iloc[:, df.columns.get_loc('Days for shipping (real)')] - \
                             df.iloc[:, df.columns.get_loc('Days for shipment (scheduled)')]

print("\n=== NULOS RESTANTES ===")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n=== SHAPE FINAL ===")
print(df.shape)

print("\n=== EJEMPLO DELAY ===")
print(df[['shipping_delay_days']].head(10))

# Guardar dataset limpio
df.to_csv('data/supply_chain_clean.csv', index=False)
print("\nDataset limpio guardado en data/supply_chain_clean.csv")