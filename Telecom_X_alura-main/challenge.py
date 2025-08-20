import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/ingridcristh/challenge2-data-science-LATAM/main/TelecomX_Data.json"
df = pd.read_json(url)

# Rename 'Churn' to 'cancelado' in the original dataframe before dropping columns
df.rename(columns={'Churn':'cancelado'}, inplace=True)

c = pd.json_normalize(df['customer'])
p = pd.json_normalize(df['phone'])
i = pd.json_normalize(df['internet'])
a = pd.json_normalize(df['account'])

# Rename columns in individual dataframes before concatenation
c.rename(columns={'gender':'genero','SeniorCitizen':'mayor65',
                  'Partner':'tienePareja','Dependents':'tieneDependentes','tenure':'mesesContrato'}, inplace=True)
p.rename(columns={'PhoneService':'servicioTelefonico','MultipleLines':'lineasMultiples'}, inplace=True)
i.rename(columns={'InternetService':'servicioInternet','OnlineSecurity':'seguridadLinea',
                  'OnlineBackup':'SoporteLinea','DeviceProtection':'proteccionDispositivos',
                  'TechSupport':'soporteTecnico','StreamingTV':'servicioTv','StreamingMovies':'servicioPeliculas'}, inplace=True)
a.rename(columns={'PaperlessBilling':'facturasElectronicas','PaymentMethod':'metodoPago',
                  'Contract':'tipoContrato','Charges.Monthly':'gastosMensuales','Charges.Total':'gastosTotales'}, inplace=True)
df.rename(columns={'customerID':'id'}, inplace=True) # Rename customerID in the original df


df = pd.concat([df.drop(columns=['customer','phone','internet','account']), c, p, i, a], axis=1)

print(df.columns) # Added this line to check column names

df['cancelado'].replace('', np.nan, inplace=True)
df['gastosTotales'].replace(' ', np.nan, inplace=True)
df.dropna(subset=['cancelado','gastosTotales'], inplace=True)

df['gastosTotales'] = df['gastosTotales'].astype(float)
df['gastosMensuales'] = df['gastosMensuales'].astype(float)

vars_str = ['id','genero','servicioInternet','tipoContrato','metodoPago']
df[vars_str] = df[vars_str].astype(str)

bools = ['cancelado','mayor65','tienePareja','tieneDependentes','facturasElectronicas']
df[bools] = df[bools].replace({'Yes':1,'No':0}).astype(int)

cats = ['lineasMultiples','seguridadLinea','SoporteLinea','proteccionDispositivos','soporteTecnico','servicioTv','servicioPeliculas']
for col in cats:
    if col in df.columns:
        df[col] = df[col].astype('category')

df['gastosDiario'] = df['gastosMensuales'] / 30

sns.set_theme(style="whitegrid", font_scale=1.1)
etiquetas = ['Permanecen','Baja']
pal = sns.color_palette("Set2", 2)

def barras_doble(s1, s2, t1, t2, ylim=None, pct=False):
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    sa = s1.reindex([0,1], fill_value=0)
    sb = s2.reindex([0,1], fill_value=0)
    r1 = ax[0].bar(etiquetas, sa, color=pal)
    for rect, val in zip(r1, sa): ax[0].text(rect.get_x()+rect.get_width()/2, val + (50 if val>0 else 0), f'{int(val):,}', ha='center', fontsize=10)
    r2 = ax[1].bar(etiquetas, sb, color=pal)
    if ylim: ax[1].set_ylim(*ylim)
    for rect, val in zip(r2, sb):
        txt = f'{val:.1f}%' if pct else f'{int(val):,}'
        ax[1].text(rect.get_x()+rect.get_width()/2, val + (1 if pct else 10), txt, ha='center', fontsize=10)
    ax[0].set_title(t1); ax[1].set_title(t2)
    plt.suptitle(f"{t1} / {t2}", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()

conteo_abs = df['cancelado'].value_counts().sort_index()
conteo_pct = df['cancelado'].value_counts(normalize=True).sort_index() * 100
barras_doble(conteo_abs, conteo_pct, "Cantidad de clientes", "Proporción de clientes", ylim=(0,100), pct=True)

if 'genero' in df.columns:
    h = df[df['genero']=='Male']['cancelado'].value_counts().sort_index()
    m = df[df['genero']=='Female']['cancelado'].value_counts().sort_index()
    barras_doble(h, m, "Cancelaciones - Hombres", "Cancelaciones - Mujeres")

if 'mayor65' in df.columns:
    df['grupoEdad'] = df['mayor65'].map({0:'< 65 años',1:'≥ 65 años'})
    menor = df[df['grupoEdad']=='< 65 años']['cancelado'].value_counts().sort_index()
    mayor = df[df['grupoEdad']=='≥ 65 años']['cancelado'].value_counts().sort_index()
    barras_doble(menor, mayor, "Cancelaciones - Menores de 65 años", "Cancelaciones - ≥ 65 años")

numerical_cols = [c for c in df.select_dtypes(include=['number']).columns if c not in ['cancelado','mayor65']]
for col in numerical_cols:
    plt.figure(figsize=(8,5))
    if col == 'gastosTotales' and 'mesesContrato' in df.columns:
        sns.violinplot(data=df[df['mesesContrato']>0], x='cancelado', y=col, palette='viridis')
    else:
        sns.violinplot(data=df, x='cancelado', y=col, palette='viridis')
    plt.title(f'Distribución de {col} por cancelado')
    plt.xlabel('cancelado'); plt.ylabel(col)
    plt.tight_layout(); plt.show()
