# Defense Q&A (Simple Answers)

1) What is the goal of this project?
- Detect and classify network attacks from traffic data.

2) Why UNSW-NB15?
- It is a standard benchmark with labeled attacks and normal traffic.

3) What models did you try?
- Logistic Regression, Decision Tree, and XGBoost.

4) Which model performed best?
- XGBoost (about 76.6% accuracy on the official test set).

5) Why do you show precision/recall/F1?
- Accuracy alone can hide poor performance on rare attack classes.

6) What does the confusion matrix show?
- Where the model confuses one class for another.

7) Why do you use raw data for testing?
- It matches real deployment and avoids hidden preprocessing bias.

8) What are limitations?
- Performance can drop on data very different from the training set.

9) How can this be improved?
- More recent data, feature tuning, and model retraining.

10) Is this system ready for production?
- It is a strong prototype, but production needs real-time data pipelines and monitoring.
