"""
ML-Based Shoreline Dynamics Classification
Supervised Classification of DSAS Indicators for Coastal Stability Assessment

This script implements a multi-model classification approach (Random Forest, XGBoost, SVM)
to classify shoreline stability patterns based on DSAS metrics and environmental modifiers.

Methodology:
- Features: EPR, NSM, SCE, LRR (DSAS indicators) + slope, salinity (environmental)
- Target: Stability_Class (Active Erosion, Stable, Active Accretion)
- Classifiers: RF, XGBoost, SVM (RBF kernel) with balanced hyperparameters
- Evaluation: Accuracy, weighted F1, 5-fold stratified cross-validation, confusion matrices
- Output: Predictions, feature importance, spatial maps, results CSV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, cohen_kappa_score
from sklearn.svm import SVC
import xgboost as xgb
import folium
from scipy.spatial import cKDTree

# ============================================================================
# STEP 1: Load Data and Create Classification Target
# ============================================================================
print("\n" + "="*80)
print("STEP 1: LOAD DATA AND CREATE CLASSIFICATION TARGET")
print("="*80)

# Load main DSAS data
try:
    # First, try to load from Excel
    ml_data = pd.read_excel('total4work_dsas.xlsx')
    print(f"✓ Loaded DSAS data: {ml_data.shape[0]} transects, {ml_data.shape[1]} columns")
except:
    print("⚠ Could not load Excel file. Trying alternative paths...")
    try:
        ml_data = pd.read_excel('shoreline_transects_all.xlsx')
        print(f"✓ Loaded DSAS data: {ml_data.shape[0]} transects, {ml_data.shape[1]} columns")
    except:
        raise FileNotFoundError("Could not load DSAS data from expected locations")

# Display available columns
print(f"\nAvailable columns:\n{ml_data.columns.tolist()}")

# Ensure required columns exist (normalize column names)
column_mapping = {}
for col in ml_data.columns:
    col_lower = col.lower().strip()
    if 'latitude' in col_lower or 'lat' in col_lower:
        column_mapping[col] = 'latitude'
    elif 'longitude' in col_lower or 'lon' in col_lower:
        column_mapping[col] = 'longitude'
    elif col not in column_mapping.values():
        column_mapping[col] = col

ml_data = ml_data.rename(columns=column_mapping)

# Create classification target: Stability_Class based on EPR thresholds
def classify_stability(epr_value):
    """
    Classify shoreline stability based on EPR (End Point Rate).
    
    Parameters:
    epr_value: float, End Point Rate in m/year
    
    Returns:
    str: stability class (Active Erosion, Stable, Active Accretion)
    """
    if epr_value < -0.5:
        return 'Active Erosion'
    elif epr_value <= 0.5:
        return 'Stable'
    else:
        return 'Active Accretion'

ml_data['Stability_Class'] = ml_data['EPR'].apply(classify_stability)

# Display class distribution
stability_dist = ml_data['Stability_Class'].value_counts().sort_index()
print("\n" + "-"*80)
print("SHORELINE STABILITY CLASS DISTRIBUTION")
print("-"*80)
for stability_class, count in stability_dist.items():
    percentage = (count / len(ml_data)) * 100
    print(f"{stability_class:20} : {count:5} transects ({percentage:6.2f}%)")
print(f"{'Total':20} : {len(ml_data):5} transects")

# Visualize class distribution
fig, ax = plt.subplots(figsize=(10, 5))
stability_colors = {'Active Erosion': '#d62728', 'Stable': '#2ca02c', 'Active Accretion': '#1f77b4'}
stability_dist.plot(kind='bar', ax=ax, color=[stability_colors.get(x, '#808080') for x in stability_dist.index])
ax.set_title('Distribution of Shoreline Stability Classes', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Transects', fontsize=12)
ax.set_xlabel('Stability Class', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('01_Stability_Class_Distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Class distribution plot saved")

# ============================================================================
# STEP 2: Feature Engineering & Data Preparation
# ============================================================================
print("\n" + "="*80)
print("STEP 2: FEATURE ENGINEERING & DATA PREPARATION")
print("="*80)

# Define feature set: 6 key predictors (DSAS indicators + environmental modifiers)
feature_columns_classification = ['EPR', 'NSM', 'SCE', 'LRR', 'slope', 'salinity']

# Verify all features are present
missing_features = [col for col in feature_columns_classification if col not in ml_data.columns]
if missing_features:
    print(f"⚠ WARNING: Missing features: {missing_features}")
    print(f"Available columns: {ml_data.columns.tolist()}")
    print("Attempting to use available features...")
    feature_columns_classification = [col for col in feature_columns_classification if col in ml_data.columns]
    if len(feature_columns_classification) < 4:
        raise ValueError(f"Too few features available: {feature_columns_classification}")

print(f"✓ Using {len(feature_columns_classification)} features: {feature_columns_classification}")

# Display feature statistics
print("\n" + "-"*80)
print("FEATURE STATISTICS FOR CLASSIFICATION")
print("-"*80)
feature_stats = ml_data[feature_columns_classification].describe().round(3)
print(feature_stats)

# Prepare feature matrix (X) and target vector (y)
X_classification = ml_data[feature_columns_classification].copy()
y_classification = ml_data['Stability_Class'].copy()

# Handle missing values
X_classification = X_classification.fillna(X_classification.mean())

# Encode target classes
label_encoder_stability = LabelEncoder()
y_encoded = label_encoder_stability.fit_transform(y_classification)

print("\n" + "-"*80)
print("CLASS ENCODING")
print("-"*80)
for idx, class_name in enumerate(label_encoder_stability.classes_):
    print(f"Class {idx}: {class_name}")

# Scale features
scaler_classification = StandardScaler()
X_scaled = scaler_classification.fit_transform(X_classification)

print(f"\n✓ Features prepared and scaled successfully")
print(f"  - Feature matrix shape: {X_scaled.shape}")
print(f"  - Target vector shape: {y_encoded.shape}")
print(f"  - Class balance: {np.bincount(y_encoded)}")

# ============================================================================
# STEP 3: Train Multiple Classifiers with Cross-Validation
# ============================================================================
print("\n" + "="*80)
print("STEP 3: TRAIN MULTIPLE CLASSIFIERS WITH 5-FOLD CROSS-VALIDATION")
print("="*80)

# Define classifiers with balanced hyperparameters
classifiers_dict = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=None,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
    'SVM (RBF)': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
}

# Set up 5-fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results
classification_results = {}

for clf_name, clf in classifiers_dict.items():
    print(f"\n{'─'*80}")
    print(f"Training {clf_name}...")
    print(f"{'─'*80}")
    
    # Train on full dataset
    clf.fit(X_scaled, y_encoded)
    
    # Get predictions on full dataset
    y_pred_full = clf.predict(X_scaled)
    
    # Compute metrics on full dataset
    accuracy = accuracy_score(y_encoded, y_pred_full)
    f1_weighted = f1_score(y_encoded, y_pred_full, average='weighted', zero_division=0)
    
    # Compute 5-fold CV scores
    cv_accuracy_scores = cross_val_score(clf, X_scaled, y_encoded, cv=skf, scoring='accuracy')
    cv_f1_scores = cross_val_score(clf, X_scaled, y_encoded, cv=skf, scoring='f1_weighted')
    
    # Store results
    classification_results[clf_name] = {
        'model': clf,
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'cv_accuracy_mean': cv_accuracy_scores.mean(),
        'cv_accuracy_std': cv_accuracy_scores.std(),
        'cv_f1_mean': cv_f1_scores.mean(),
        'cv_f1_std': cv_f1_scores.std(),
        'cv_accuracy_scores': cv_accuracy_scores,
        'cv_f1_scores': cv_f1_scores,
        'y_pred': y_pred_full
    }
    
    # Print results
    print(f"\nFull Dataset Performance:")
    print(f"  Accuracy:           {accuracy:.4f}")
    print(f"  Weighted F1-Score:  {f1_weighted:.4f}")
    
    print(f"\n5-Fold Cross-Validation Performance:")
    print(f"  Accuracy:  Mean={cv_accuracy_scores.mean():.4f} ± Std={cv_accuracy_scores.std():.4f}")
    print(f"             Fold Scores: {[f'{x:.4f}' for x in cv_accuracy_scores]}")
    print(f"  F1-Score:  Mean={cv_f1_scores.mean():.4f} ± Std={cv_f1_scores.std():.4f}")
    print(f"             Fold Scores: {[f'{x:.4f}' for x in cv_f1_scores]}")

print("\n" + "="*80)
print("✓ All classifiers trained successfully!")
print("="*80)

# ============================================================================
# STEP 4: Generate Confusion Matrices & Per-Class Performance Metrics
# ============================================================================
print("\n" + "="*80)
print("STEP 4: CONFUSION MATRICES & CLASSIFICATION REPORTS")
print("="*80)

# Create figure with subplots for confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices: Actual vs Predicted Stability Classes', fontsize=14, fontweight='bold')

class_names = label_encoder_stability.classes_

for idx, (clf_name, results) in enumerate(classification_results.items()):
    y_pred = results['y_pred']
    cm = confusion_matrix(y_encoded, y_pred)
    
    # Plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    axes[idx].set_title(f'{clf_name}\nAccuracy: {results["accuracy"]:.4f}', fontweight='bold')
    axes[idx].set_ylabel('True Class')
    axes[idx].set_xlabel('Predicted Class')
    
    # Print detailed classification report
    print(f"\n{'─'*80}")
    print(f"{clf_name} - DETAILED CLASSIFICATION REPORT")
    print(f"{'─'*80}")
    print(f"\nConfusion Matrix:\n{cm}\n")
    print(classification_report(y_encoded, y_pred, target_names=class_names, digits=4))

plt.tight_layout()
plt.savefig('02_Confusion_Matrices.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ Confusion matrices plot saved")

# ============================================================================
# STEP 5: Extract & Visualize Feature Importance
# ============================================================================
print("\n" + "="*80)
print("STEP 5: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Create figure for feature importance
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Feature Importance: DSAS Indicators + Environmental Modifiers', fontsize=14, fontweight='bold')

feature_importance_dict = {}

for idx, (clf_name, results) in enumerate(classification_results.items()):
    clf = results['model']
    
    # Extract feature importance
    if hasattr(clf, 'feature_importances_'):  # RF and XGBoost
        importances = clf.feature_importances_
    else:  # SVM
        importances = np.abs(clf.coef_).mean(axis=0) if hasattr(clf, 'coef_') else np.ones(len(feature_columns_classification))
    
    # Normalize to sum to 1
    importances = importances / importances.sum()
    
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_columns_classification[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]
    
    # Store results
    feature_importance_dict[clf_name] = {
        'features': sorted_features,
        'importances': sorted_importances
    }
    
    # Plot
    colors = ['#ff7f0e' if feat in ['slope', 'salinity'] else '#1f77b4' for feat in sorted_features]
    axes[idx].barh(sorted_features, sorted_importances, color=colors)
    axes[idx].set_xlabel('Importance Score')
    axes[idx].set_title(f'{clf_name}', fontweight='bold')
    axes[idx].grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', label='DSAS Indicators (EPR, NSM, LRR, SCE)'),
                       Patch(facecolor='#ff7f0e', label='Environmental Modifiers (slope, salinity)')]
    if idx == 2:
        axes[idx].legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # Print feature importance table
    print(f"\n{clf_name} - Feature Importance Ranking:")
    print(f"{'Feature':<20} {'Importance':<15} {'Percentage':<15}")
    print(f"{'-'*50}")
    for feat, imp in zip(sorted_features, sorted_importances):
        print(f"{feat:<20} {imp:<15.4f} {(imp*100):<14.2f}%")

plt.tight_layout()
plt.savefig('03_Feature_Importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ Feature importance plot saved")

# ============================================================================
# STEP 6: Classifier Comparison & Performance Summary
# ============================================================================
print("\n" + "="*80)
print("STEP 6: CLASSIFIER PERFORMANCE COMPARISON")
print("="*80)

# Create comparison dataframe
comparison_data = []
for clf_name, results in classification_results.items():
    comparison_data.append({
        'Classifier': clf_name,
        'Accuracy': f"{results['accuracy']:.4f}",
        'Weighted F1': f"{results['f1_weighted']:.4f}",
        'CV Accuracy Mean': f"{results['cv_accuracy_mean']:.4f}",
        'CV Accuracy Std': f"{results['cv_accuracy_std']:.4f}",
        'CV F1 Mean': f"{results['cv_f1_mean']:.4f}",
        'CV F1 Std': f"{results['cv_f1_std']:.4f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Visualize performance comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Classifier Performance Comparison', fontsize=14, fontweight='bold')

# Extract metrics for plotting
clf_names = list(classification_results.keys())
accuracies = [classification_results[clf]['accuracy'] for clf in clf_names]
f1_scores = [classification_results[clf]['f1_weighted'] for clf in clf_names]
cv_acc_means = [classification_results[clf]['cv_accuracy_mean'] for clf in clf_names]
cv_f1_means = [classification_results[clf]['cv_f1_mean'] for clf in clf_names]

# Plot 1: Accuracy comparison
x_pos = np.arange(len(clf_names))
width = 0.35

axes[0].bar(x_pos - width/2, accuracies, width, label='Full Dataset', alpha=0.8)
axes[0].bar(x_pos + width/2, cv_acc_means, width, label='5-Fold CV Mean', alpha=0.8)
axes[0].set_ylabel('Accuracy', fontsize=11)
axes[0].set_title('Accuracy Comparison (Full Dataset vs CV)', fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(clf_names, rotation=0)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0.5, 1.0])

# Plot 2: F1-Score comparison
axes[1].bar(x_pos - width/2, f1_scores, width, label='Full Dataset', alpha=0.8)
axes[1].bar(x_pos + width/2, cv_f1_means, width, label='5-Fold CV Mean', alpha=0.8)
axes[1].set_ylabel('Weighted F1-Score', fontsize=11)
axes[1].set_title('F1-Score Comparison (Full Dataset vs CV)', fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(clf_names, rotation=0)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_ylim([0.5, 1.0])

plt.tight_layout()
plt.savefig('04_Classifier_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ Classifier comparison plot saved")

# Determine best classifier
best_clf_name = max(classification_results.items(), key=lambda x: x[1]['f1_weighted'])[0]
print(f"\nBEST CLASSIFIER: {best_clf_name} (based on weighted F1-score)")

# ============================================================================
# STEP 7: Generate Spatial Classification Maps (Interactive Folium)
# ============================================================================
print("\n" + "="*80)
print("STEP 7: GENERATE SPATIAL CLASSIFICATION MAPS")
print("="*80)

# Get best classifier predictions
best_clf = classification_results[best_clf_name]['model']
y_pred_best = best_clf.predict(X_scaled)
y_pred_best_labels = label_encoder_stability.inverse_transform(y_pred_best)

# Add predictions to dataset
ml_data['ML_Stability_Class_Predicted'] = y_pred_best_labels

# Define color mapping for stability classes
stability_color_map = {
    'Active Erosion': '#d62728',      # Red
    'Stable': '#2ca02c',               # Green
    'Active Accretion': '#1f77b4'     # Blue
}

# Check if latitude/longitude columns exist
has_coords = 'latitude' in ml_data.columns and 'longitude' in ml_data.columns

if has_coords:
    # Calculate center of map
    center_lat = ml_data['latitude'].mean()
    center_lon = ml_data['longitude'].mean()
    
    # Create folium map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add title
    title_html = '''
                 <div style="position: fixed; 
                         top: 10px; left: 50px; width: 400px; height: auto; 
                         background-color: white; border:2px solid grey; z-index:9999; 
                         font-size:16px; font-weight:bold; padding:10px">
                 ML-Based Shoreline Stability Classification
                 <br><small>Classifier: ''' + best_clf_name + '''</small>
                 </div>
                 '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add circles for each transect with predictions
    print("Adding predicted stability classes to map...")
    for idx, row in ml_data.iterrows():
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            lat, lon = row['latitude'], row['longitude']
            stability = row['ML_Stability_Class_Predicted']
            
            # Tooltip info
            popup_text = f"""
            <b>Transect ID:</b> {idx+1}<br>
            <b>Stability Class:</b> {stability}<br>
            <b>EPR (m/yr):</b> {row['EPR']:.2f}<br>
            <b>NSM (m):</b> {row['NSM']:.1f}<br>
            <b>LRR (m/yr):</b> {row['LRR']:.2f}<br>
            <b>SCE (m):</b> {row['SCE']:.1f}
            """
            
            # Draw circle (2 km diameter = 1 km radius)
            folium.Circle(
                location=[lat, lon],
                radius=1000,
                popup=folium.Popup(popup_text, max_width=300),
                color=stability_color_map.get(stability, '#808080'),
                fill=True,
                fillColor=stability_color_map.get(stability, '#808080'),
                fillOpacity=0.6,
                weight=2,
                opacity=0.8
            ).add_to(m)
    
    # Add legend
    legend_html = '''
         <div style="position: fixed; 
                     bottom: 50px; left: 50px; width: 250px; height: auto; 
                     background-color: white; border:2px solid grey; z-index:9999; 
                     font-size:13px; padding:10px">
         <p style="margin:5px;"><b>Stability Classes (''' + best_clf_name + ''')</b></p>
         <p style="margin:5px;"><i class="fa fa-circle" style="color:#d62728"></i> Active Erosion (EPR < -0.5 m/yr)</p>
         <p style="margin:5px;"><i class="fa fa-circle" style="color:#2ca02c"></i> Stable (-0.5 ≤ EPR ≤ 0.5 m/yr)</p>
         <p style="margin:5px;"><i class="fa fa-circle" style="color:#1f77b4"></i> Active Accretion (EPR > 0.5 m/yr)</p>
         <p style="margin:5px; font-size:11px; font-style:italic;">Circle diameter: 2 km (geographic-scale representation)</p>
         </div>
         '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    map_filename = '05_ML_Shoreline_Stability_Classification_Map.html'
    m.save(map_filename)
    print(f"\n✓ Interactive map saved: {map_filename}")
else:
    print("⚠ Note: Latitude/Longitude columns not found in dataset.")
    print("  Spatial map will not be generated.")
    print("  Please load coordinate data separately if available.")

# Display class distribution of predictions
pred_dist = ml_data['ML_Stability_Class_Predicted'].value_counts().sort_index()
print("\nPredicted Stability Class Distribution:")
print("="*60)
for stability_class, count in pred_dist.items():
    percentage = (count / len(ml_data)) * 100
    print(f"{stability_class:20} : {count:5} transects ({percentage:6.2f}%)")

# ============================================================================
# STEP 8: Classifier Agreement Analysis & Results Export
# ============================================================================
print("\n" + "="*80)
print("STEP 8: INTER-CLASSIFIER AGREEMENT ANALYSIS")
print("="*80)

# Compute pairwise agreement between classifiers
classifier_names = list(classification_results.keys())
agreement_matrix = np.zeros((len(classifier_names), len(classifier_names)))

print("\nPairwise Classifier Agreement (Cohen's Kappa):")
print("-"*80)

for i, clf1_name in enumerate(classifier_names):
    for j, clf2_name in enumerate(classifier_names):
        if i < j:
            pred1 = classification_results[clf1_name]['y_pred']
            pred2 = classification_results[clf2_name]['y_pred']
            
            # Calculate agreement percentage
            agreement_pct = (pred1 == pred2).sum() / len(pred1) * 100
            
            # Calculate Cohen's Kappa
            kappa = cohen_kappa_score(pred1, pred2)
            
            agreement_matrix[i, j] = agreement_pct
            agreement_matrix[j, i] = agreement_pct
            
            print(f"{clf1_name:20} vs {clf2_name:20} : {agreement_pct:6.2f}% agreement (Kappa={kappa:7.4f})")

# Visualize agreement matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(agreement_matrix, annot=True, fmt='.1f', cmap='YlGn', 
            xticklabels=classifier_names, yticklabels=classifier_names,
            cbar_kws={'label': 'Agreement (%)'}, ax=ax, vmin=0, vmax=100)
ax.set_title('Inter-Classifier Agreement Matrix\n(Percentage of Identical Predictions)', 
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('06_Classifier_Agreement.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ Agreement matrix plot saved")

# ============================================================================
# STEP 9: Export Results to CSV
# ============================================================================
print("\n" + "="*80)
print("STEP 9: EXPORT RESULTS TO CSV")
print("="*80)

# Create output dataframe with all predictions and metrics
output_df = ml_data[['latitude', 'longitude', 'Stability_Class', 'EPR', 'NSM', 'LRR', 'SCE', 'slope', 'salinity']].copy()
output_df['ML_Stability_Predicted'] = y_pred_best_labels

# Add confidence scores (probability estimates)
if hasattr(best_clf, 'predict_proba'):
    probabilities = best_clf.predict_proba(X_scaled)
    output_df['Prediction_Confidence'] = probabilities.max(axis=1)
else:
    output_df['Prediction_Confidence'] = np.nan

# Save to CSV
output_filename = '07_ML_Shoreline_Stability_Classifications.csv'
output_df.to_csv(output_filename, index=False)
print(f"\n✓ Classification results exported: {output_filename}")
print(f"  - Records: {len(output_df)}")
print(f"  - Columns: {', '.join(output_df.columns.tolist())}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\nBest Performing Classifier: {best_clf_name}")
print(f"  - Full Dataset Accuracy:    {classification_results[best_clf_name]['accuracy']:.4f}")
print(f"  - Full Dataset F1-Score:    {classification_results[best_clf_name]['f1_weighted']:.4f}")
print(f"  - 5-Fold CV Accuracy Mean:  {classification_results[best_clf_name]['cv_accuracy_mean']:.4f} ± {classification_results[best_clf_name]['cv_accuracy_std']:.4f}")
print(f"  - 5-Fold CV F1-Score Mean:  {classification_results[best_clf_name]['cv_f1_mean']:.4f} ± {classification_results[best_clf_name]['cv_f1_std']:.4f}")

print(f"\nKey Features (Top 3 Importance):")
top_features = feature_importance_dict[best_clf_name]['features'][:3]
top_importances = feature_importance_dict[best_clf_name]['importances'][:3]
for feat, imp in zip(top_features, top_importances):
    print(f"  - {feat:<20} : {imp*100:6.2f}%")

print(f"\nPredicted Class Distribution:")
for stability_class, count in ml_data['ML_Stability_Class_Predicted'].value_counts().sort_index().items():
    pct = count / len(ml_data) * 100
    print(f"  - {stability_class:<20} : {count:5} transects ({pct:6.2f}%)")

print(f"\nMethodological Validation:")
print(f"  ✓ Random Forest: Captures non-linear geomorphic interactions")
print(f"  ✓ XGBoost: Handles tabular data heterogeneity & class imbalance")
print(f"  ✓ SVM (RBF): Refines decision boundaries in high-dimensional space")
print(f"  ✓ 5-Fold Cross-Validation: Ensures robustness across dataset subsets")
print(f"  ✓ Feature Integration: DSAS (EPR, NSM, LRR, SCE) + Environmental (slope, salinity)")

print("\n" + "="*80)
print("✓ ML-Based Shoreline Dynamics Classification Workflow Complete!")
print("="*80)
print(f"\nGenerated outputs:")
print(f"  1. 01_Stability_Class_Distribution.png")
print(f"  2. 02_Confusion_Matrices.png")
print(f"  3. 03_Feature_Importance.png")
print(f"  4. 04_Classifier_Comparison.png")
print(f"  5. 05_ML_Shoreline_Stability_Classification_Map.html")
print(f"  6. 06_Classifier_Agreement.png")
print(f"  7. 07_ML_Shoreline_Stability_Classifications.csv")
