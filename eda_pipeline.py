import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)

def create_sample_eda_dataset(filepath='/workspace/scratch/coaster_db_sample.csv'):
    """
    Generates a sample dataset modeled after the Roller Coaster EDA dataset
    used in Rob Mulla's Exploratory Data Analysis tutorial.
    """
    np.random.seed(42)
    n_samples = 250
    
    coasters = [f'Coaster_{i}' for i in range(1, n_samples + 1)]
    parks = np.random.choice(['Six Flags', 'Cedar Point', 'Universal Studios', 'Disneyland', 'Busch Gardens'], size=n_samples)
    material_type = np.random.choice(['Steel', 'Wooden', 'Hybrid'], size=n_samples, p=[0.6, 0.3, 0.1])
    speed_mph = np.round(np.random.normal(60, 15, size=n_samples), 1)
    height_ft = np.round(speed_mph * 1.8 + np.random.normal(0, 10, size=n_samples), 1)
    length_ft = np.round(np.random.normal(3000, 800, size=n_samples), 1)
    inversions = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=n_samples, p=[0.4, 0.1, 0.15, 0.15, 0.1, 0.05, 0.03, 0.02])
    year_introduced = np.random.randint(1980, 2024, size=n_samples)
    rating = np.round(np.random.uniform(3.0, 5.0, size=n_samples), 2)
    
    # Introduce some missing values to demonstrate data cleaning
    speed_mph[np.random.choice(n_samples, 10, replace=False)] = np.nan
    height_ft[np.random.choice(n_samples, 8, replace=False)] = np.nan
    
    df = pd.DataFrame({
        'Coaster_Name': coasters,
        'Location_Park': parks,
        'Material_Type': material_type,
        'Speed_mph': speed_mph,
        'Height_ft': height_ft,
        'Length_ft': length_ft,
        'Inversions_Count': inversions,
        'Year_Introduced': year_introduced,
        'Rating_Score': rating
    })
    
    df.to_csv(filepath, index=False)
    print(f"Sample dataset generated and saved to '{filepath}'.")
    return filepath

def run_eda_pipeline(csv_path):
    print("="*60)
    print("STEP 1: DATA UNDERSTANDING")
    print("="*60)
    df = pd.read_csv(csv_path)
    
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print("--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Data Types & Non-Null Counts ---")
    print(df.info())
    
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    print("\n--- Missing Value Counts ---")
    print(df.isna().sum())

    print("\n" + "="*60)
    print("STEP 2: DATA PREPARATION & CLEANING")
    print("="*60)
    
    # 1. Selecting relevant subset of columns
    keep_cols = ['Coaster_Name', 'Location_Park', 'Material_Type', 'Speed_mph', 
                 'Height_ft', 'Length_ft', 'Inversions_Count', 'Year_Introduced', 'Rating_Score']
    df_clean = df[keep_cols].copy()
    
    # 2. Renaming columns for consistency
    df_clean = df_clean.rename(columns={
        'Location_Park': 'Park',
        'Speed_mph': 'Speed_MPH',
        'Height_ft': 'Height_FT',
        'Length_ft': 'Length_FT',
        'Inversions_Count': 'Inversions',
        'Rating_Score': 'Rating'
    })
    
    # 3. Handling missing values (median imputation for numerical features)
    df_clean['Speed_MPH'] = df_clean['Speed_MPH'].fillna(df_clean['Speed_MPH'].median())
    df_clean['Height_FT'] = df_clean['Height_FT'].fillna(df_clean['Height_FT'].median())
    
    # 4. Checking and removing duplicate coaster names
    print(f"Duplicate Coasters Count: {df_clean.duplicated(subset=['Coaster_Name']).sum()}")
    df_clean = df_clean.drop_duplicates(subset=['Coaster_Name']).reset_index(drop=True)
    print(f"Cleaned Dataset Shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

    print("\n" + "="*60)
    print("STEP 3: FEATURE DISTRIBUTIONS (UNIVARIATE ANALYSIS)")
    print("="*60)
    
    # Plot 1: Top 10 Parks by Coaster Count
    plt.figure(figsize=(8, 4))
    park_counts = df_clean['Park'].value_counts().head(10)
    ax1 = park_counts.plot(kind='bar', title='Top Amusement Parks by Coaster Count', color='teal')
    ax1.set_xlabel('Park Name')
    ax1.set_ylabel('Number of Coasters')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('/workspace/scratch/eda_park_distribution.png', dpi=150)
    plt.close()
    print("Saved '/workspace/scratch/eda_park_distribution.png'")
    
    # Plot 2: Speed Distribution (Histogram + KDE)
    plt.figure(figsize=(8, 4))
    sns.histplot(df_clean['Speed_MPH'], kde=True, bins=20, color='coral')
    plt.title('Distribution of Coaster Speed (MPH)', fontsize=12, fontweight='bold')
    plt.xlabel('Speed (MPH)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('/workspace/scratch/eda_speed_distribution.png', dpi=150)
    plt.close()
    print("Saved '/workspace/scratch/eda_speed_distribution.png'")

    print("\n" + "="*60)
    print("STEP 4: FEATURE RELATIONSHIPS (BIVARIATE ANALYSIS)")
    print("="*60)
    
    # Plot 3: Scatter Plot - Speed vs. Height by Material Type
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Speed_MPH', y='Height_FT', hue='Material_Type', data=df_clean, s=70, alpha=0.8)
    plt.title('Coaster Speed vs. Height by Material Type', fontsize=12, fontweight='bold')
    plt.xlabel('Speed (MPH)')
    plt.ylabel('Height (FT)')
    plt.tight_layout()
    plt.savefig('/workspace/scratch/eda_speed_vs_height.png', dpi=150)
    plt.close()
    print("Saved '/workspace/scratch/eda_speed_vs_height.png'")
    
    # Plot 4: Correlation Heatmap
    plt.figure(figsize=(7, 5))
    num_cols = ['Speed_MPH', 'Height_FT', 'Length_FT', 'Inversions', 'Year_Introduced', 'Rating']
    corr_matrix = df_clean[num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Features', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/workspace/scratch/eda_correlation_heatmap.png', dpi=150)
    plt.close()
    print("Saved '/workspace/scratch/eda_correlation_heatmap.png'")

    print("\n" + "="*60)
    print("STEP 5: ASKING & ANSWERING A DATA QUESTION (INSIGHTS)")
    print("="*60)
    print("Question: Which park has the highest average coaster speed for coasters introduced since 2000?")
    
    recent_coasters = df_clean[df_clean['Year_Introduced'] >= 2000]
    park_speed = recent_coasters.groupby('Park')['Speed_MPH'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    print("\nAverage Speed by Park (Coasters built >= 2000):")
    print(park_speed)
    
    # Plot 5: Average Speed by Park
    plt.figure(figsize=(8, 4))
    park_speed['mean'].plot(kind='barh', color='purple')
    plt.title('Average Coaster Speed by Park (Coasters built >= 2000)', fontsize=12, fontweight='bold')
    plt.xlabel('Average Speed (MPH)')
    plt.ylabel('Park')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('/workspace/scratch/eda_park_avg_speed.png', dpi=150)
    plt.close()
    print("Saved '/workspace/scratch/eda_park_avg_speed.png'")

if __name__ == '__main__':
    dataset_file = create_sample_eda_dataset()
    run_eda_pipeline(dataset_file)
