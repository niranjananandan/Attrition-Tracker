import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

def analyze_employee_risk(emp_id: str, dataframe: pd.DataFrame) -> dict:
    """
    Analyzes an individual employee's data to extract all driving factors.
    Returns a dictionary with 'risk' and 'retention' keys, each containing a list of 
    normalized dictionaries with reason, percentage, and detail.
    """
    # 1. Look up employee
    emp_df = dataframe[dataframe['Employee ID'] == emp_id]
    if emp_df.empty:
        return {'risk': [], 'retention': []}
    
    # 2. Re-create the model and explainer
    features = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'JobSatisfaction']
    available_features = [f for f in features if f in dataframe.columns]
    if not available_features:
        available_features = dataframe.select_dtypes(include=[np.number]).columns.tolist()[:4]
        
    X = dataframe[available_features].fillna(0)
    y = dataframe['Predicted_Risk_Percentage'] >= 50
    
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    
    # 3. Extract the employee's features
    emp_row = emp_df[available_features].fillna(0)
    
    # 4. Get SHAP values
    shap_vals = explainer.shap_values(emp_row)
    if isinstance(shap_vals, list):
        sv_array = shap_vals[1][0, :] # positive class
    else:
        sv_array = shap_vals[0, :, 1] if len(shap_vals.shape) == 3 else shap_vals[0, :]
        
    # 5. Separate risk and retention factors
    risk_factors = []
    retention_factors = []
    
    for i, feature in enumerate(available_features):
        val = sv_array[i]
        actual_val = emp_row.iloc[0][feature]
        
        # Determine clean reason
        reason = ''.join([' ' + char if char.isupper() else char for char in feature]).strip()
        
        # Format the actual value cleanly
        if 'Income' in feature:
            val_str = f"${actual_val:,.0f}"
        else:
            val_str = str(actual_val)
            
        if val > 0:
            risk_factors.append({
                'feature': feature,
                'shap_value': float(val),
                'reason': reason,
                'actual_value': val_str
            })
        elif val < 0:
            retention_factors.append({
                'feature': feature,
                'shap_value': float(abs(val)), # absolute value for magnitude
                'reason': reason,
                'actual_value': val_str
            })
            
    # Sort by highest impact magnitude
    risk_factors = sorted(risk_factors, key=lambda x: x['shap_value'], reverse=True)
    retention_factors = sorted(retention_factors, key=lambda x: x['shap_value'], reverse=True)
    
    def process_list(factor_list, type_name):
        total_impact = sum(f['shap_value'] for f in factor_list)
        results = []
        for f in factor_list:
            percentage = 0 if total_impact == 0 else round((f['shap_value'] / total_impact) * 100)
            
            if type_name == 'risk':
                detail = f"The employee's {f['reason'].lower()} is {f['actual_value']}, which is pushing them towards leaving the company."
            else:
                detail = f"The employee's {f['reason'].lower()} is {f['actual_value']}, which is a strong factor keeping them satisfied and at the company."
            
            results.append({
                'reason': f['reason'],
                'percentage': percentage,
                'detail': detail
            })
            
        # Adjust rounding errors so it sums exactly to 100%
        if results and total_impact > 0:
            total_pct = sum(r['percentage'] for r in results)
            if total_pct != 100 and total_pct > 0:
                results[0]['percentage'] += (100 - total_pct)
        return results

    return {
        'risk': process_list(risk_factors, 'risk'),
        'retention': process_list(retention_factors, 'retention')
    }