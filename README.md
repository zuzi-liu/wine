# Wine Decision Support System 🍷

**Preference-Based Recommendation and Quality Optimization Using Machine Learning**

## Project Overview
This project builds a decision system for wine selection and design using a real-world dataset of wine chemical properties. It combines:

* Predictive modeling (Ridge Regression, Random Forest)
* Preference-based recommendation
* Constrained optimization

The goal is to answer: How can we recommend or design wines that best match user preferences while maintaining high quality?

## Dataset

This project uses the UCI Wine Quality Dataset, which contains physicochemical measurements of Portuguese red and white wines.

### Source Files
* `winequality-red.csv`
* `winequality-white.csv`

These datasets were combined into a single dataset with an additional feature indicating wine type.

### Features

Each wine is described by the following variables:

| Feature              | Description                                          |
| -------------------- | ---------------------------------------------------- |
| fixed acidity        | Non-volatile acids contributing to taste             |
| volatile acidity     | Acetic acid content (vinegar-like taste)             |
| citric acid          | Adds freshness and flavor                            |
| residual sugar       | Remaining sugar after fermentation (sweetness proxy) |
| chlorides            | Salt content                                         |
| free sulfur dioxide  | Prevents microbial growth                            |
| total sulfur dioxide | Total SO₂ level                                      |
| density              | Related to sugar and alcohol content                 |
| pH                   | Acidity level                                        |
| sulphates            | Wine preservative and antioxidant                    |
| alcohol              | Alcohol percentage                                   |
| type                 | Red (0) or White (1) wine                            |

### Target Variable

| Variable | Description                               |
| -------- | ----------------------------------------- |
| quality  | Sensory quality score (integer from 0–10) |


## Data Processing

The dataset required minimal cleaning. The following steps were applied:

* Combined red and white datasets
* Added a `type` feature to distinguish wine categories
* Standardized column names to `snake_case`
* Verified no missing values
* Scaled features for model training

## Project Objectives

This project is structured around three main components:

### 1. Predictive Modeling

We train machine learning models to estimate wine quality:

$$
\hat{quality} = f(x)
$$

Models used:

* Ridge Regression (baseline, interpretable)
* Random Forest Regressor (nonlinear, higher performance)

These models learn relationships between chemical properties and perceived quality.


### 2. Preference-Based Recommendation

Users can input preferences such as:

* sweetness (dry → sweet)
* alcohol strength
* acidity level

These preferences are mapped to dataset features (e.g., residual sugar → sweetness), and wines are scored based on:

$$
score = \alpha \cdot \text{preference match} + (1 - \alpha) \cdot \text{predicted quality}
$$

The system returns the top-k wines that best match user preferences.


### 3. Optimization 

We extend the model into a decision-making framework:

$$
\max_x \quad \alpha \cdot \text{preference match}(x) + (1 - \alpha)\cdot \hat{quality}(x)
$$

subject to:

* feature bounds (based on dataset ranges)
* optional constraints (e.g., minimum quality threshold)

This produces an **“optimal wine profile”** that balances user preferences and predicted quality.


## Modeling Assumptions

* Chemical properties are treated as controllable inputs
* Preference mappings (e.g., sugar → sweetness) are **proxies, not exact measures
* The model is predictive, not causal
* Optimization results should be interpreted as decision-support guidance, not guaranteed outcomes


## Project Structure

```bash
wine_decision_tool/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── wine_decision_tool/
│   ├── data.py
│   ├── model.py
│   ├── recommend.py
│   ├── optimize.py
│
├── tests/
│
├── notebooks/
│   └── eda.ipynb
│
├── README.md
├── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Example workflow:

```python
from wine_decision_tool.model import train_model
from wine_decision_tool.recommend import recommend_wines

model = train_model(data)

recommend_wines(
    sweetness=0.2,
    alcohol=0.8,
    acidity=0.4
)
```


## Testing

This project includes unit tests covering:

* data loading and preprocessing
* model training and prediction
* recommendation logic
* optimization constraints

Run tests with:

```bash
pytest
```

## 📈 Future Improvements

* Incorporate uncertainty (robust optimization)
* Add more interpretable preference mappings
* Build a user interface (CLI or web app)
* Integrate external wine datasets
