
One more VERY important thing

UNSW-NB15 has some rare categorical values.

If users upload unseen values like:

service = unknown_service

your encoder may crash.

To avoid this:

Use:

OneHotEncoder(handle_unknown='ignore')

This is a VERY important professional touch.

“The XGBoost model achieved 76% multiclass classification accuracy on the UNSW-NB15 dataset while demonstrating strong detection capability for dominant attack categories. Performance limitations were largely influenced by class imbalance and the complexity of distinguishing similar network attack behaviors.”

Strong defense strategy

Say something like:

“The discrepancy between the official evaluation accuracy and the inference accuracy may be attributed to differences in data distribution, class composition, and attack complexity between datasets. The official UNSW-NB15 testing set presented a more challenging and balanced multiclass environment, whereas the inference dataset appeared to contain more predictable traffic patterns and fewer difficult attack classes.” 

1. Demo CSV

Use:
demo_attack_mix.csv

Include:

multiple attack categories
realistic traffic
balanced examples
clean formatting

Recommended size:

30–100 rows

Perfect for live presentation.

For demonstration purposes, a controlled inference dataset was prepared from the official testing distribution to showcase the deployed intrusion detection system in a stable and reproducible manner.

Final recommendation

YES:

prepare a controlled demo CSV
derive it from the official testing set
use it only for demonstration
keep official metrics separate


What you SHOULD do for the demo dataset
1. Randomize rows

Yes — definitely.

This prevents:

grouped attack patterns
ordered classes
repetitive predictions

Example:

demo_df = demo_df.sample(frac=1, random_state=42)

This shuffles the rows.

Good practice.

2. Remove unnecessary/problematic rows

Yes — moderately.

For example:

duplicate rows
corrupted rows
NaN-heavy rows
malformed records

can be removed from the DEMO dataset.

Why?

Because your goal during defense is:

stable inference
smooth presentation
clear predictions

NOT stress-testing the app live.

3. Balance the attack classes somewhat

This is VERY smart.

If your demo dataset is:

95% Normal traffic

then your predictions become boring.

Instead:

intentionally include multiple attack categories.

Example:

10 Normal
10 Exploits
10 Fuzzers
10 DoS

This creates a visually impressive multiclass demonstration.

What you should NOT do
Do NOT:
fabricate fake records
manually alter labels
manipulate results
over-clean the official evaluation dataset

That becomes academically risky.

VERY IMPORTANT:
Your official metrics should remain untouched

Your:

75–76% accuracy
confusion matrix
precision/recall/F1

should come from the REAL official testing pipeline.

Not from a curated demo file.

This distinction is important.

Recommended workflow
Step 1 — Use official test dataset
test_df = pd.read_csv("UNSW_NB15_testing-set.csv")
Step 2 — Sample balanced attacks
demo_df = (
    test_df.groupby("attack_cat")
    .sample(10, replace=True, random_state=42)
)
Step 3 — Shuffle
demo_df = demo_df.sample(frac=1, random_state=42)
Step 4 — Remove target columns
demo_input = demo_df.drop(
    columns=["label", "attack_cat"]
)
Step 5 — Save CSV
demo_input.to_csv(
    "demo_attack_mix.csv",
    index=False
)

This is the ideal approach.

Why this is academically safe

Because:

data still comes from official dataset
no fake values introduced
no label manipulation
preprocessing consistency maintained

You are simply preparing:

a controlled demonstration subset.

Which is completely acceptable.

VERY IMPORTANT DEFENSE POINT

If asked:

“Why did you prepare a separate demo dataset?”

You can say:

“The demonstration dataset was created as a controlled subset of the official testing distribution to ensure stable live inference, multiclass visibility, and reproducible deployment behavior during presentation.”

That is a strong answer.