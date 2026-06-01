# =============================================================================
# TRAVEL INSURANCE PREDICTION — PREPROCESSING & EDA
# Case Study | Machine Learning LAB | MVSR Engineering College
# Subject Code: U21PC682IT | Academic Year: 2025-2026
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

sns.set_theme(style="whitegrid", palette="Blues_d")
plt.rcParams['figure.dpi'] = 120

# =============================================================================
# SECTION 1 — LOAD DATASET
# =============================================================================

df = pd.read_csv('travel_insurance.csv')

print("=" * 60)
print("SECTION 1: DATASET OVERVIEW")
print("=" * 60)
print(f"\nShape (rows, cols): {df.shape}")
print(f"\nColumn Names:\n{df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")

# =============================================================================
# SECTION 2 — MISSING VALUES & DUPLICATES
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: MISSING VALUES & DUPLICATES")
print("=" * 60)

print(f"\nMissing values per column:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# =============================================================================
# SECTION 3 — PREPROCESSING
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: PREPROCESSING")
print("=" * 60)

# 3.1 Drop duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\n[3.1] Duplicates removed : {before - len(df)} | Shape now: {df.shape}")

# 3.2 Remove impossible Age and Duration values
df = df[(df['Age'] > 0) & (df['Age'] <= 100)]
df = df[df['Duration'] >= 0]
print(f"[3.2] After outlier row removal  : {df.shape}")

# 3.3 Fill missing Gender
df['Gender'] = df['Gender'].fillna('Unknown')
print(f"[3.3] Missing values after fill  : {df.isnull().sum().sum()}")

# 3.4 Encode target variable
df['Claim_Binary'] = (df['Claim'] == 'Yes').astype(int)
print(f"\n[3.4] Target encoding done.")
print(f"      Claim distribution:\n{df['Claim'].value_counts()}")
print(f"      Claim %:\n{(df['Claim'].value_counts(normalize=True)*100).round(2)}")

# 3.5 Label encode categorical columns
le = LabelEncoder()
cat_cols = ['Agency', 'Agency Type', 'Distribution Channel',
            'Product Name', 'Destination', 'Gender']
for col in cat_cols:
    df[col + '_enc'] = le.fit_transform(df[col])
print(f"\n[3.5] Label encoding done for: {cat_cols}")

# 3.6 Standard scale numerical columns
scaler = StandardScaler()
scale_cols = ['Duration', 'Net Sales', 'Commision (in value)', 'Age']
df[['Duration_scaled', 'NetSales_scaled',
    'Commission_scaled', 'Age_scaled']] = scaler.fit_transform(df[scale_cols])
print(f"[3.6] Standard scaling done for: {scale_cols}")

print(f"\nFinal clean dataset shape: {df.shape}")

# =============================================================================
# SECTION 4 — STATISTICAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4: STATISTICAL SUMMARY")
print("=" * 60)

print("\nDescriptive statistics (raw numerical columns):")
print(df[['Duration', 'Net Sales', 'Commision (in value)', 'Age']].describe().round(2).to_string())

print("\nCorrelation with target (Claim_Binary):")
num_df = df[['Duration', 'Net Sales', 'Commision (in value)', 'Age', 'Claim_Binary']]
corr = num_df.corr()
print(corr['Claim_Binary'].sort_values(ascending=False).round(4))

print("\nTop 10 Destinations by volume:")
print(df['Destination'].value_counts().head(10))

print("\nAgency Type distribution:")
print(df['Agency Type'].value_counts())

print("\nDistribution Channel:")
print(df['Distribution Channel'].value_counts())

# =============================================================================
# SECTION 5 — EDA VISUALISATIONS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: EDA VISUALISATIONS")
print("=" * 60)

# ── Figure 1: Claim Distribution ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['Claim'].value_counts()
colors = ['#2E75B6', '#C00000']
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', width=0.5)
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f'{v:,}\n({v / len(df) * 100:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Figure 1: Claim Distribution (Target Variable)', fontweight='bold', fontsize=12)
ax.set_xlabel('Claim Status')
ax.set_ylabel('Count')
ax.set_ylim(0, 62000)
plt.tight_layout()
plt.savefig('fig1_claim_distribution.png', bbox_inches='tight')
plt.show()
print("[Fig 1] Claim distribution saved.")

# ── Figure 2: Age Distribution by Claim ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for label, color in [('No', '#2E75B6'), ('Yes', '#C00000')]:
    df[df['Claim'] == label]['Age'].plot(
        kind='hist', bins=30, alpha=0.65, ax=ax,
        label=f'Claim = {label}', color=color, edgecolor='white')
ax.set_title('Figure 2: Age Distribution by Claim Status', fontweight='bold', fontsize=12)
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('fig2_age_distribution.png', bbox_inches='tight')
plt.show()
print("[Fig 2] Age distribution saved.")

# ── Figure 3: Trip Duration Boxplot ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
data_box = [
    df[df['Claim'] == 'No']['Duration'].clip(0, 300),
    df[df['Claim'] == 'Yes']['Duration'].clip(0, 300)
]
bp = ax.boxplot(data_box, tick_labels=['No Claim', 'Claim'], patch_artist=True,
                medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], ['#2E75B6', '#C00000']):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_title('Figure 3: Trip Duration vs Claim Status (clipped @ 300 days)',
             fontweight='bold', fontsize=12)
ax.set_ylabel('Duration (days)')
plt.tight_layout()
plt.savefig('fig3_duration_boxplot.png', bbox_inches='tight')
plt.show()
print("[Fig 3] Duration boxplot saved.")

# ── Figure 4: Net Sales Boxplot ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
data_ns = [
    df[df['Claim'] == 'No']['Net Sales'].clip(0, 300),
    df[df['Claim'] == 'Yes']['Net Sales'].clip(0, 300)
]
bp2 = ax.boxplot(data_ns, tick_labels=['No Claim', 'Claim'], patch_artist=True,
                 medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp2['boxes'], ['#2E75B6', '#C00000']):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_title('Figure 4: Net Sales vs Claim Status', fontweight='bold', fontsize=12)
ax.set_ylabel('Net Sales (SGD)')
plt.tight_layout()
plt.savefig('fig4_netsales_boxplot.png', bbox_inches='tight')
plt.show()
print("[Fig 4] Net sales boxplot saved.")

# ── Figure 5: Agency Type & Distribution Channel ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x='Agency Type', hue='Claim', data=df,
              palette=['#2E75B6', '#C00000'], ax=axes[0])
axes[0].set_title('Figure 5a: Agency Type vs Claim', fontweight='bold')
axes[0].legend(title='Claim')

sns.countplot(x='Distribution Channel', hue='Claim', data=df,
              palette=['#2E75B6', '#C00000'], ax=axes[1])
axes[1].set_title('Figure 5b: Distribution Channel vs Claim', fontweight='bold')
axes[1].legend(title='Claim')

plt.tight_layout()
plt.savefig('fig5_agency_channel.png', bbox_inches='tight')
plt.show()
print("[Fig 5] Agency & channel plots saved.")

# ── Figure 6: Correlation Heatmap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='Blues', ax=ax,
            linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.8})
ax.set_title('Figure 6: Correlation Heatmap (Numerical Features)',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('fig6_correlation_heatmap.png', bbox_inches='tight')
plt.show()
print("[Fig 6] Correlation heatmap saved.")

# ── Figure 7: Top 10 Destinations by Claim Rate ───────────────────────────────
dest_claim = df.groupby('Destination')['Claim_Binary'].agg(['sum', 'count'])
dest_claim['rate'] = dest_claim['sum'] / dest_claim['count'] * 100
top10 = dest_claim.nlargest(10, 'rate').reset_index()

print("\nTop 10 Destinations by Claim Rate:")
print(top10[['Destination', 'sum', 'count', 'rate']].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.barh(top10['Destination'], top10['rate'],
               color='#2E75B6', edgecolor='white')
for bar, v in zip(bars, top10['rate']):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f'{v:.1f}%', va='center', fontsize=9)
ax.set_title('Figure 7: Top 10 Destinations by Claim Rate (%)',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Claim Rate (%)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('fig7_destination_claimrate.png', bbox_inches='tight')
plt.show()
print("[Fig 7] Destination claim rate chart saved.")

# =============================================================================
# SECTION 6 — PRODUCT LEVEL ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 6: PRODUCT-LEVEL CLAIM RATE ANALYSIS")
print("=" * 60)

prod_claim = df.groupby('Product Name')['Claim_Binary'].agg(['sum', 'count'])
prod_claim['claim_rate_%'] = (prod_claim['sum'] / prod_claim['count'] * 100).round(2)
prod_claim.columns = ['Claims', 'Total Policies', 'Claim Rate (%)']
print(prod_claim.sort_values('Claim Rate (%)', ascending=False).head(10).to_string())

# =============================================================================
# SECTION 7 — FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 7: FINAL SUMMARY")
print("=" * 60)

print(f"""
Dataset          : travel_insurance.csv
Raw rows         : 63,326
After cleaning   : {len(df):,}
  - Duplicates removed   : 8,042
  - Invalid Age removed  : 652
  - Negative Duration    : 5

Target variable  : Claim (Yes/No)
  - No Claim     : {(df['Claim']=='No').sum():,}  ({(df['Claim']=='No').mean()*100:.2f}%)
  - Claim        : {(df['Claim']=='Yes').sum():,}   ({(df['Claim']=='Yes').mean()*100:.2f}%)

Top Correlations with Claim:
  Net Sales              : +0.133
  Commision (in value)   : +0.096
  Duration               : +0.070
  Age                    : -0.016

Key Findings:
  1. Severe class imbalance (1.67% positive class)
  2. Annual plans have 10-12% claim rate (6-7x average)
  3. Longer trips = higher claim probability
  4. Singapore is highest-volume high-risk destination
  5. Age is NOT a meaningful predictor

Figures saved:
  fig1_claim_distribution.png
  fig2_age_distribution.png
  fig3_duration_boxplot.png
  fig4_netsales_boxplot.png
  fig5_agency_channel.png
  fig6_correlation_heatmap.png
  fig7_destination_claimrate.png
""")