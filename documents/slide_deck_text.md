# Slide Deck Text (Bullets + Speaker Notes)

## Slide 1: Title
Bullets:
- AI-Based Intrusion Detection System (UNSW-NB15)
- Final Year Project
- Name, Department, Date

Speaker notes:
- "This project builds a machine learning system that detects and classifies network attacks using the UNSW-NB15 dataset. I will show the problem, method, results, and a short demo."

## Slide 2: Problem Statement
Bullets:
- Networks face frequent cyber attacks
- Manual inspection is slow and error-prone
- Need faster, automated detection

Speaker notes:
- "In real networks, attacks happen quickly. Human analysis alone is too slow, so we need automatic detection." 

## Slide 3: Project Goal
Bullets:
- Detect and classify attacks from traffic data
- Provide a simple dashboard for results
- Evaluate performance with clear metrics

Speaker notes:
- "The goal is not just detection, but classification of attack types, and a UI that makes results easy to understand."

## Slide 4: Dataset
Bullets:
- UNSW-NB15 benchmark dataset
- 10 classes: Normal + 9 attack types
- Official split used for fair evaluation

Speaker notes:
- "UNSW-NB15 is a standard benchmark with labeled traffic. We keep the official train/test split to avoid bias."

## Slide 5: Method Overview (Simple)
Bullets:
- Clean and format data
- Convert text fields to numbers
- Scale numeric features
- Add traffic-behavior features

Speaker notes:
- "We clean the data, encode categories, scale values, and add a few ratio features that capture traffic behavior."

## Slide 6: Models Compared
Bullets:
- Logistic Regression
- Decision Tree
- XGBoost (best)

Speaker notes:
- "We compared three models to balance simplicity and performance. XGBoost performed best." 

## Slide 7: Evaluation Metrics
Bullets:
- Accuracy
- Precision, Recall, F1
- Confusion Matrix

Speaker notes:
- "Accuracy alone can hide poor performance on rare attacks, so we also report precision, recall, and F1." 

## Slide 8: Results (Key Numbers)
Bullets:
- Best accuracy: 76.60% (XGBoost)
- Macro precision/recall/F1 reported
- Balanced view of rare classes

Speaker notes:
- "XGBoost reaches about 76.6% accuracy on the official test set. Macro scores show how well it handles rare attacks." 

## Slide 9: Demo Walkthrough
Bullets:
- Run Streamlit app
- Upload test file
- View charts and metrics

Speaker notes:
- "I will upload the test dataset, and the app will show metrics, the confusion matrix, and prediction distribution." 

## Slide 10: Limitations
Bullets:
- Research dataset, not live traffic
- Performance depends on data quality
- Distribution shift can reduce accuracy

Speaker notes:
- "This is a strong prototype, but real-world traffic can differ. Performance can drop if the data changes." 

## Slide 11: Future Work
Bullets:
- Real-time data pipeline
- Continuous retraining
- Security monitoring integration

Speaker notes:
- "Next steps include real-time ingestion and retraining to keep models updated." 

## Slide 12: Conclusion
Bullets:
- End-to-end IDS prototype built
- Good performance on benchmark
- Ready for further development

Speaker notes:
- "The project achieves its goal: an IDS prototype that detects and classifies attacks, with clear evaluation and a usable UI." 
