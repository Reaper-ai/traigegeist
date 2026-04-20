from pathlib import Path
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

RANDOM_STATE = 42

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def apply_cyclical_encoding(df):

    """ Applies cyclical encoding to time-related features in the dataframe."""

    print("Applying Cyclical Encoding to Time Features...")

    # 1. Clean and convert arrival_time (HHMM integer) to a continuous hour scale (0 to 23.99)
    # Example: 1430 -> 14 hours + (30/60) minutes = 14.5
    df['arrival_time'] = np.where(df['arrival_time'] < 0, np.nan, df['arrival_time'])
        
    # Extract hours and minutes
    hours = np.floor_divide(df['arrival_time'], 100)
    minutes = np.mod(df['arrival_time'], 100)
        
    # Create the continuous hour feature
    df['arrival_hour_float'] = hours + (minutes / 60.0)
        
    # Shift overlap flag (06:00-08:00, 18:00-20:00)
    df['is_shift_change'] = (((df['arrival_hour_float'] >= 6) & (df['arrival_hour_float'] < 8))
        | ((df['arrival_hour_float'] >= 18) & (df['arrival_hour_float'] < 20))
        ).astype(int)

    # 2. Define the exact maximum values for a full cycle
    cycle_maxes = {
        'visit_month': 12.0,
        'day_of_week': 7.0,
        'arrival_hour_float': 24.0
    }

    # 3. Apply the Sine and Cosine Transformations
    for col, max_val in cycle_maxes.items():
        if col in df.columns:
            # Calculate the angle on the circle
            angle = 2 * np.pi * df[col] / max_val
            
            # Create the Sin and Cos features
            df[f'{col}_sin'] = np.sin(angle)
            df[f'{col}_cos'] = np.cos(angle)

    # Weekend/weeknight interaction
    max_dow = df['day_of_week'].max()
    weekend_days = [5, 6] if max_dow <= 6 else [6, 7]
    is_weekend = df['day_of_week'].isin(weekend_days)
    df['is_weekend_night'] = (is_weekend & (df['arrival_hour_float'] >= 18)).astype(int)

    # 4. Drop the original linear time columns to prevent multicollinearity
    cols_to_drop = ['visit_month', 'day_of_week', 'arrival_time', 'arrival_hour_float']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    print(f"Dataframe shape after cyclical encoding: {df.shape}")
    print("\nSample of the new Time Features:")
    return df
    
def news2_score(df):
    """Calculates an approximate NEWS2 score based on available vital sign features in the dataframe."""
    # NEWS2-style score (approximate)
    news2 = pd.Series(0, index=df.index, dtype=float)

    sbp = df["sys_bp"]
    hr = df["heart_rate"]
    t = df["temp"]
    sp = df["spo2"]
    rr = df["resp_rate"]

    news2 += np.select([t <= 95.0,(t > 95.0) & (t <= 96.8),(t > 96.8) & (t <= 100.4),(t > 100.4) & (t <= 102.2),t > 102.2,], [3, 1, 0, 1, 2], default=0)
    news2 += np.select([rr <= 8,(rr >= 9) & (rr <= 11), (rr >= 12) & (rr <= 20),(rr >= 21) & (rr <= 24),rr >= 25,], [3, 1, 0, 2, 3], default=0)
    news2 += np.select([sp <= 91,(sp >= 92) & (sp <= 93),(sp >= 94) & (sp <= 95),sp >= 96,], [3, 2, 1, 0], default=0)
    news2 += np.select([sbp <= 90,(sbp >= 91) & (sbp <= 100),(sbp >= 101) & (sbp <= 110),(sbp >= 111) & (sbp <= 219), sbp >= 220,], [3, 2, 1, 0, 3], default=0)
    news2 += np.select([hr <= 40, (hr >= 41) & (hr <= 50), (hr >= 51) & (hr <= 90),(hr >= 91) & (hr <= 110),(hr >= 111) & (hr <= 130),hr >= 131,], [3, 1, 0, 1, 2, 3], default=0)

    return news2

def apply_clinical_ratios(df):
    """ Adds clinically relevant ratios and interaction features to the dataframe."""

    print("Adding clinical ratios and vital missingness...")

    df["shock_index"] = df["heart_rate"] / df["sys_bp"].replace(0, np.nan)
    df["shock_index"] = df["shock_index"] * np.where(df["age"] >= 65, 1.2, 1.0)

    df["map"] = (df["sys_bp"] + 2 * df["dias_bp"]) / 3
    df["pulse_pressure"] = df["sys_bp"] - df["dias_bp"]

    df["age_hr_interaction"] = df["age"] * df["heart_rate"]

    df["resp_spo2_ratio"] = df["resp_rate"] / df["spo2"].replace(0, np.nan)

    df["elderly_tachy"] = ((df["age"] >= 65) & (df["heart_rate"] > 100)).astype(int)

    history_cols = [c for c in df.columns if any(k in c for k in "hist_")]
    if history_cols:
        hist_numeric = df[history_cols].apply(pd.to_numeric, errors="coerce")
        df["history_count"] = hist_numeric.fillna(0).sum(axis=1)


    df["news2_score"] = news2_score(df)

    return df

def drop_leaky_features(df):
    """drops features that are likely to be leaky (i.e., those that are direct proxies for the target variable)."""

    cols_to_drop = ["intervention_iv_fluids","vitals_during_visit","wait_time_minutes"]
   
    df.drop(columns=cols_to_drop, inplace=True)

def exclude_bias_features(df):
    """excludes features that could introduce bias into the model (e.g., demographic features that may lead to unfair predictions)."""

    bias_columns = {"residence", "region", "race", "no_payment", "insurance"}
    df.drop(columns=[c for c in df.columns if c in bias_columns], inplace=True)

def convert_categorical(df):
    print("Converting object / string features to categorical...")

    categorical_cols = df.select_dtypes(include=['object', 'str']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    print(f"Dataframe shape after converting categoricals: {df.shape}")
    return df


# Define Target (Grouped ESI)
def map_esi(val):
    if val in (1, 2): return 0
    if val == 3: return 1
    return 2 if val in (4, 5) else np.nan