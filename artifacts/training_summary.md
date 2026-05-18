# Training Summary

Best model: Random Forest

Model comparison:
 accuracy  precision   recall       f1  roc_auc               model
 0.809794   0.386431 0.685864 0.494340 0.829005       Random Forest
 0.755855   0.330377 0.780105 0.464174 0.842489 Logistic Regression
 0.706175   0.286807 0.785340 0.420168 0.806880       Decision Tree

Top feature importance:
                            feature  importance
                             tenure    0.204163
                     MonthlyCharges    0.169004
                       TotalCharges    0.118626
        InternetService_Fiber optic    0.078008
            Contract_Month-to-month    0.040462
                InternetService_DSL    0.028704
                  Contract_Two year    0.028082
                 InternetService_No    0.021325
StreamingMovies_No internet service    0.019394
    TechSupport_No internet service    0.016839

Business insights were saved to artifacts/reports/business_insights.md