# Defense Overview (Plain English)

## What problem are we solving?
Networks are attacked every day. This project detects suspicious traffic and classifies the type of attack so a defender can respond quickly.

## What is our solution?
We built a system that learns patterns from a well-known dataset (UNSW-NB15). It then predicts whether new network traffic is normal or one of several attack types.

## How it works (simple steps)
1. Read the network data file.
2. Clean the data (fix missing values and formats).
3. Convert text fields into numbers and scale everything to the same range.
4. Add a few extra features that capture traffic behavior.
5. Use a trained model to predict the attack type.

## What the demo shows
- You upload a dataset.
- The app predicts classes and shows charts.
- If labels exist, you see accuracy, precision, recall, and F1.

## Why this matters
- Faster detection means faster response.
- Classifying attack types helps prioritize action.

## Limitations (honest but safe)
- Results depend on the quality of input data.
- This is a research dataset, not live traffic.
- Performance can drop if data differs from the training distribution.
