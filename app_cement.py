# Cement Mortar Strength Predictor App: Training, Prediction & Scientific Reporting Interface

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, RepeatedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
import optuna

# --- App Setup ---
st.set_page_config(layout='wide')
st.title("Cement Mortar Strength Predictor")
st.subheader("Training, Prediction, and Scientific Reporting")

# --- Tabs ---
tab_train, tab_predict, tab_report = st.tabs(["Training 🔧", "Prediction 🤖", "Report 📊"])

# --- Shared Utilities ---
@st.cache_data
def load_and_clean(file):
    df = pd.read_csv(file)
    df = df.apply(lambda c: c.str.strip().str.replace('\xa0','') if c.dtype=='object' else c)
    return df.apply(pd.to_numeric, errors='coerce')

@st.cache_data
def clean_split(df, target, drop_cols, test_size):
    df = df.dropna(subset=[target, '2CS', 'Loi']).dropna(thresh=0.5*len(df), axis=1).dropna()
    X = df.drop(columns=[target] + drop_cols)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=42)

@st.cache_data
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler

# --- Repeated CV Helper ---
def repeated_cv(model, X, y, folds=3, repeats=2):
    rk = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
    cv = cross_validate(model, X, y, cv=rk,
                        scoring=['neg_mean_absolute_error','r2'], n_jobs=-1)
    mae = -cv['test_neg_mean_absolute_error']
    r2 = cv['test_r2']
    return mae.mean(), mae.std(), r2.mean(), r2.std()

# --- Hyperparameter Spaces ---
rf_params = lambda t: {
    'n_estimators': t.suggest_int('n_estimators', 100, 500, step=50),
    'max_depth': t.suggest_int('max_depth', 5, 50, step=5),
    'min_samples_split': t.suggest_int('min_samples_split', 2, 10),
    'min_samples_leaf': t.suggest_int('min_samples_leaf', 1, 5),
    'max_features': t.suggest_categorical('max_features', ['sqrt','log2', None])
}

xgb_params = lambda t: {
    'n_estimators': t.suggest_int('n_estimators', 100, 500, step=50),
    'max_depth': t.suggest_int('max_depth', 3, 20),
    'learning_rate': t.suggest_float('learning_rate', 1e-3, 0.1),
    'subsample': t.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': t.suggest_float('colsample_bytree', 0.6, 1.0),
    'gamma': t.suggest_float('gamma', 0, 5),
    'reg_alpha': t.suggest_float('reg_alpha', 1e-5, 1),
    'reg_lambda': t.suggest_float('reg_lambda', 1e-5, 1)
}

def build_trial_nn(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_units = trial.suggest_categorical('hidden_units', [32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2)
    model = Sequential([InputLayer(shape=(X_train_s.shape[1],))])
    for _ in range(n_layers):
        model.add(Dense(hidden_units, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
    return model

nn_params = lambda t: {
    'build_fn': lambda: build_trial_nn(t),
    'epochs': 100,
    'batch_size': t.suggest_categorical('batch_size', [16, 32, 64]),
    'validation_split': 0.2,
    'callbacks': [tf.keras.callbacks.EarlyStopping(patience=5)],
    'verbose': 0
}

# --- Optuna Objectives ---
def objective_cv_mae(cls, paramf, X, y):
    def obj(trial):
        params = paramf(trial)
        model = cls(**params)
        scores = cross_validate(model, X, y, cv=3,
                                scoring='neg_mean_absolute_error', n_jobs=-1)
        return -scores['test_score'].mean()
    return obj


def objective_test_mae(cls, paramf, X_tr, y_tr, X_te, y_te):
    def obj(trial):
        params = paramf(trial)
        model = cls(**params)
        model.fit(X_tr, y_tr)
        return mean_absolute_error(y_te, model.predict(X_te))
    return obj

# --- Session Store ---
if 'models' not in st.session_state:
    st.session_state['models'] = {}
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = pd.DataFrame()

# --- Training Tab ---
with tab_train:
    with st.container(border=True):
        st.header("Training & Optimization 🔧")
        st.markdown("Upload data, then run Baselines or Hyperparameter Optimization.")
        data_file = st.file_uploader("Upload CSV", type=['csv'])
        left, right = st.columns(2)
        if data_file:
            df = load_and_clean(data_file)
            with left:
                st.subheader("Data Description")
                with st.expander("ℹ️ Data Cleaning & Preprocessing Steps", expanded=False):
                    st.markdown("""
                    **Overview of Preprocessing Operations:**
                    - **Whitespace & Non-breaking Space Removal**: All string columns are stripped of leading/trailing spaces and non-breaking spaces (`\\xa0`).
                    - **Type Conversion**: Attempts to convert all data to numeric format; non-convertible entries become `NaN`.
                    - **NaN Filtering**:
                        - Rows where the target (`28CS`) or auxiliary variable (`2CS` and 'LOI') is missing are removed.
                        - Columns with more than 50% missing data are dropped.
                    - **Feature & Target Split**: Drops selected columns and isolates the target (`28CS`) for regression.
                    - **Scaling**: StandardScaler is applied to center and scale feature data based on training statistics.
                    """)
                    desc = df.describe().T
                    desc['dtype'] = df.dtypes
                    st.dataframe(desc)

            with right:
                st.subheader("Feature Selection")
                drop_cols = st.multiselect("Columns to drop", df.columns.tolist(), default=['7CS','56CS'])
                target = '28CS'
                st.subheader("Hold-out test set percentage")
                test_size = st.slider("Test split fraction", 0.1, 0.5, 0.2)
            if target and target in df.columns:
                X_train, X_test, y_train, y_test = clean_split(df, target, drop_cols, test_size)
                X_train_s, X_test_s, scaler = scale_data(X_train, X_test)
                st.session_state['scaler'] = scaler
            with left:
                with st.expander("📊 Cleaned Data Summary", expanded=False):
                    st.markdown(f"""
                    **Post-cleaning Dataset Characteristics:**
                    - **Target Variable**: `{target}`
                    - **Dropped Columns**: `{', '.join(drop_cols) if drop_cols else 'None'}`
                    - **Training Set Size**: {X_train.shape[0]} samples × {X_train.shape[1]} features
                    - **Test Set Size**: {X_test.shape[0]} samples × {X_test.shape[1]} features
                    - **Feature Scaling**: StandardScaler applied to training data and used on test data.
                    - **Missing Values**: All rows with `NaN` values were removed; dataset is now complete.
                    """)
                    st.dataframe(pd.DataFrame(X_train, columns=X_train.columns).describe().T)

            # Baseline naming
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.subheader("Baseline Models")
                    baseline_prefix = st.text_input("Baseline run name prefix", value='baseline')
                
                    if st.button("Run Baselines"):
                        ests = {
                            # 'RF': RandomForestRegressor(random_state=42),
                            # 'XGB': XGBRegressor(objective='reg:squarederror', random_state=42),
                            #'GB': GradientBoostingRegressor(random_state=42),
                            'LR': LinearRegression(),
                            'SVR': SVR()
                        }
                        def build_nn():
                            m = Sequential([InputLayer(shape=(X_train_s.shape[1],)),
                                            Dense(64,'relu'),Dropout(0.2),Dense(64,'relu'),Dropout(0.2),Dense(1)])
                            m.compile('adam','mse')
                            return m
                        ests['NN'] = KerasRegressor(build_fn=build_nn, epochs=150, batch_size=32,
                                                    validation_split=0.2,
                                                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)],
                                                    verbose=0)
                        metrics = {}
                        for name, est in ests.items():
                            m, ms, r, rs = repeated_cv(est, X_train_s, y_train)
                            est.fit(X_train_s, y_train)
                            test_r2 = r2_score(y_test, est.predict(X_test_s))
                            key = f"{baseline_prefix}_{name}"
                            metrics[key] = {'CV MAE': m, 'CV MAE std': ms, 'CV R2': r, 'CV R2 std': rs, 'Test R2': test_r2}
                            st.session_state['models'][key] = est
                        dfm = pd.DataFrame(metrics).T
                        st.session_state['metrics'] = dfm
                        st.table(dfm)
            with col2:
                with st.container(border=True):
                    st.subheader("Hyperparameter Optimization")
                    sel = st.selectbox("Model to optimize", ['RF','XGB','GB','SVR','NN'])
                    opt_name = st.text_input("Name for optimized model", value=f"{sel}_opt")
                    objective_choice = st.radio("Objective", ['CV MAE','Test MAE'])
                    trials = st.slider('Number of Optuna trials', 5, 250, 50, step=5)

                    # Map models to their param functions
                    spaces = {
                        'RF': (rf_params, RandomForestRegressor),
                        'XGB': (xgb_params, XGBRegressor),
                        'GB': (rf_params, GradientBoostingRegressor),
                        'SVR': (lambda t: {'C': t.suggest_float('C', 0.1, 10.0)}, SVR),
                        'NN': (nn_params, KerasRegressor)
                    }
                    base_paramf, cls = spaces[sel]
                    st.markdown("**Adjust Hyperparameter Ranges:**")
                    # Build dynamic paramf
                    if sel == 'RF':
                        ne_min, ne_max = st.slider('n_estimators', 10, 1000, (100, 500), step=10)
                        ne_step = st.number_input('n_estimators step', min_value=1, value=50)
                        md_min, md_max = st.slider('max_depth', 1, 100, (5, 50))
                        ms_split_min, ms_split_max = st.slider('min_samples_split', 2, 20, (2, 10))
                        ms_leaf_min, ms_leaf_max = st.slider('min_samples_leaf', 1, 10, (1, 5))
                        max_feat = st.multiselect('max_features', ['sqrt','log2', None], default=['sqrt','log2', None])
                        def paramf(trial):
                            return {
                                'n_estimators': trial.suggest_int('n_estimators', ne_min, ne_max, step=ne_step),
                                'max_depth': trial.suggest_int('max_depth', md_min, md_max),
                                'min_samples_split': trial.suggest_int('min_samples_split', ms_split_min, ms_split_max),
                                'min_samples_leaf': trial.suggest_int('min_samples_leaf', ms_leaf_min, ms_leaf_max),
                                'max_features': trial.suggest_categorical('max_features', max_feat)
                            }
                    elif sel == 'XGB':
                        ne_min, ne_max = st.slider('n_estimators', 50, 1000, (100, 500), step=50)
                        lr_min, lr_max = st.select_slider('learning_rate', options=[0.001,0.005,0.01,0.05,0.1,0.2], value=(0.001, 0.1))
                        md_min, md_max = st.slider('max_depth', 1, 20, (3, 10))
                        def paramf(trial):
                            return {
                                'n_estimators': trial.suggest_int('n_estimators', ne_min, ne_max, step=50),
                                'max_depth': trial.suggest_int('max_depth', md_min, md_max),
                                'learning_rate': trial.suggest_float('learning_rate', lr_min, lr_max),
                                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                                'gamma': trial.suggest_float('gamma', 0, 5),
                                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1),
                                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1)
                            }
                    elif sel == 'GB':
                        ne_min, ne_max = st.slider('n_estimators', 50, 1000, (100, 500), step=50)
                        md_min, md_max = st.slider('max_depth', 1, 20, (3, 10))
                        def paramf(trial):
                            return {
                                'n_estimators': trial.suggest_int('n_estimators', ne_min, ne_max, step=50),
                                'max_depth': trial.suggest_int('max_depth', md_min, md_max),
                                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
                            }
                    elif sel == 'SVR':
                        c_min, c_max = st.slider('C range', 0.1, 100.0, (0.1, 10.0))
                        def paramf(trial):
                            return {'C': trial.suggest_float('C', c_min, c_max)}
                    else:
                        layer_min = st.number_input('min layers', 1, 5, value=1)
                        layer_max = st.number_input('max layers', layer_min, 10, value=3)
                        hu_opts = st.multiselect('hidden units', [32,64,128,256], default=[32,64,128])
                        def paramf(trial):
                            return nn_params(trial)

                    if st.button("Run Optimization", key=f"opt_{sel}"):
                        study = optuna.create_study(direction='minimize')
                        prog = st.progress(0); stat = st.empty(); history = []
                        def callback(study, trial):
                            history.append(study.best_value)
                            prog.progress(len(history)/trials)
                            stat.text(f"Trial {len(history)}/{trials} best: {study.best_value:.3f}")
                        objective = objective_cv_mae(cls, paramf, X_train_s, y_train) if objective_choice=='CV MAE' else objective_test_mae(cls, paramf, X_train_s, y_train, X_test_s, y_test)
                        study.optimize(objective, n_trials=int(trials), callbacks=[callback])
                        st.write("**Best params:**")
                        st.json(study.best_params)
                        best = study.best_params
                        if sel != 'NN':
                            model_opt = cls(**best, random_state=42) if 'random_state' in cls().get_params() else cls(**best)
                        else:
                            model_opt = KerasRegressor(build_fn=lambda: build_trial_nn(study.best_trial), epochs=best.get('epochs',50), batch_size=best.get('batch_size',32), validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)], verbose=0)
                        model_opt.fit(X_train_s, y_train)
                        st.session_state['models'][opt_name] = model_opt
                        st.success(f"Optimized model '{opt_name}' stored.")

with tab_predict:
    st.header("Make Predictions 🤖")

    if st.session_state.get('models'):
        name = st.selectbox("📌 Select Trained Model", list(st.session_state['models'].keys()))
        model = st.session_state['models'][name]
        scaler = st.session_state['scaler']

        st.markdown("### 🧾 Input Feature Values")
        with st.expander("✏️ Customize Inputs", expanded=True):
            input_cols = st.columns(2)  # Two-column layout
            inputs = {}
            for idx, feature in enumerate(X_train.columns):
                with input_cols[idx % 2]:
                    default = float(X_train[feature].mean())
                    inputs[feature] = st.number_input(f"{feature}", value=default, format="%.3f")

        if st.button("🚀 Run Prediction"):
            x = np.array([list(inputs.values())])
            x_s = scaler.transform(x)
            pred = model.predict(x_s)
            
            st.success("✅ Prediction completed!")
            st.markdown("### 🎯 Predicted Strength:")
            st.metric(label="**28-Day Compressive Strength (28CS)**", value=f"{pred[0]:.3f}", delta="MPa")
    else:
        st.info("⚠️ No trained models found. Please run Baselines or Optimization first.")


# --- Reporting Tab ---
def compute_test_metrics_all(models, X_test, y_test):
    test_mae, test_rmse = {}, {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        test_mae[name] = mean_absolute_error(y_test, y_pred)
        test_rmse[name] = mean_squared_error(y_test, y_pred, squared=False)
    return test_mae, test_rmse

def compute_cv_rmse_all(models, X_train, y_train, n_splits=5, n_repeats=1, random_state=42):
    cv_rmse = {}
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    for name, model in models.items():
        neg_mse = cross_validate(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)['test_score']
        cv_rmse[name] = np.sqrt(-neg_mse).mean()
    return cv_rmse

def build_comparison_dfs(metrics_full, test_mae, test_rmse, cv_rmse):
    df_mae = pd.DataFrame({'CV MAE': metrics_full['CV MAE'], 'Test MAE': pd.Series(test_mae)})
    df_r2  = metrics_full[['CV R2','Test R2']]
    df_rmse = pd.DataFrame({'CV RMSE': pd.Series(cv_rmse), 'Test RMSE': pd.Series(test_rmse)})
    return df_mae, df_r2, df_rmse

def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        return pd.Series(model.feature_importances_, index=feature_names).sort_values()
    return None

with tab_report:
    st.header("📊 Scientific Reporting")

    # Load metrics and models
    base_metrics = st.session_state.get('metrics', pd.DataFrame())
    full_metrics = base_metrics.copy()
    models = st.session_state.get('models', {})

    # Append optimized models if missing
    for name, model in models.items():
        if name not in full_metrics.index:
            try:
                m, ms, r, rs = repeated_cv(model, X_train_s, y_train)
                test_r2 = r2_score(y_test, model.predict(X_test_s))
                full_metrics.loc[name] = {
                    'CV MAE': m, 'CV MAE std': ms,
                    'CV R2': r, 'CV R2 std': rs,
                    'Test R2': test_r2
                }
            except Exception:
                continue
    st.session_state['metrics'] = full_metrics

    # --- Comparison Charts ---
    with st.expander("📈 Model Comparison Charts", expanded=False):
        if st.button("Show Comparison Charts") and not full_metrics.empty:
            test_mae, test_rmse = compute_test_metrics_all(models, X_test_s, y_test)
            cv_rmse = compute_cv_rmse_all(models, X_train_s, y_train)
            df_mae, df_r2, df_rmse = build_comparison_dfs(full_metrics, test_mae, test_rmse, cv_rmse)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("MAE Comparison")
                fig, ax = plt.subplots(); df_mae.plot(kind='bar', ax=ax); ax.set_ylabel("MAE"); st.pyplot(fig)
            with col1:
                st.subheader("R² Comparison")
                fig, ax = plt.subplots(); df_r2.plot(kind='bar', ax=ax); ax.set_ylabel("R²"); st.pyplot(fig)
            with col2:
                st.subheader("RMSE Comparison")
                fig, ax = plt.subplots(); df_rmse.plot(kind='bar', ax=ax); ax.set_ylabel("RMSE"); st.pyplot(fig)

            # Error Distributions Across Models
            st.subheader("📌 Residual Distributions Across Models")
            residual_data = []
            for name, model in models.items():
                try:
                    preds = model.predict(X_test_s)
                    errs = y_test - preds
                    temp_df = pd.DataFrame({'Model': name, 'Residual': errs})
                    residual_data.append(temp_df)
                except Exception:
                    continue

            if residual_data:
                residual_df = pd.concat(residual_data)
                import seaborn as sns
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x='Model', y='Residual', data=residual_df, ax=ax)
                ax.axhline(0, color='gray', linestyle='--')
                ax.set_title("Residual Distribution by Model")
                with col2:
                    st.pyplot(fig)
        elif full_metrics.empty:
            st.warning("No metrics to display yet.")

    st.markdown("---")

    # --- Detailed Model Analysis ---
    st.subheader("🔍 Detailed Model Metrics & Diagnostics")
    sel_name = st.selectbox("Select model for detailed report", full_metrics.index.tolist())
    
    if sel_name:
        sel_model = models.get(sel_name)
        st.write("### 📋 Metrics Summary")
        st.dataframe(full_metrics.loc[sel_name].to_frame().T)

        col_fi, col_sc = st.columns(2)

        # Feature Importances
        with col_fi:
            fi = get_feature_importance(sel_model, X_train.columns)
            st.subheader("📌 Feature Importances")
            if fi is not None:
                fig, ax = plt.subplots()
                fi.plot(kind='barh', ax=ax, color='teal')
                ax.set_xlabel('Importance')
                st.pyplot(fig)
            else:
                st.info("Feature importances not available for this model.")

        # Prediction vs Actual ±10%
        with col_sc:
            st.subheader("📊 Prediction vs Actual ±10%")
            y_pred_test = sel_model.predict(X_test_s)
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred_test, alpha=0.6)
            mn = min(y_test.min(), y_pred_test.min())
            mx = max(y_test.max(), y_pred_test.max())
            ax.plot([mn, mx], [mn, mx], 'k--')
            ax.plot([mn, mx], [1.1*mn, 1.1*mx], 'r:', label='+10%')
            ax.plot([mn, mx], [0.9*mn, 0.9*mx], 'r:', label='-10%')
            ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
            ax.set_xlabel('Actual 28CS'); ax.set_ylabel('Predicted 28CS')
            ax.legend()
            st.pyplot(fig)
            st.markdown(f"**Test R²:** `{r2_score(y_test, y_pred_test):.3f}`")

        # Residual Analysis
        st.subheader("🧮 Residual Analysis")
        residuals = y_test - y_pred_test
        col_res1, col_res2 = st.columns(2)

        # Residuals vs Predicted
        with col_res1:
            fig, ax = plt.subplots()
            ax.scatter(y_pred_test, residuals, alpha=0.6)
            ax.axhline(0, color='black', linestyle='--')
            ax.set_xlabel("Predicted 28CS")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Predicted")
            st.pyplot(fig)

        # Residual Histogram
        with col_res2:
            fig, ax = plt.subplots()
            ax.hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.8)
            ax.axvline(residuals.mean(), color='red', linestyle='--', label='Mean Error')
            ax.set_title("Error Distribution (Residuals)")
            ax.set_xlabel("Error (Actual - Predicted)")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)
    else:
        st.info("Select a trained model above to view detailed reporting.")

