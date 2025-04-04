
# **Iris Flower Classification Using Logistic Regression & Decision Tree Classifier**

## **Overview**
This project implements **Logistic Regression** and **Decision Tree Classifier** to classify Iris flower species based on their **sepal and petal measurements**. The dataset is extracted from an SQLite database and analyzed using visualization techniques before training the models.

---

## **Dataset**
The dataset used is the classic **Iris dataset**, which consists of:
- **150 samples** of Iris flowers.
- **4 features**: Sepal Length, Sepal Width, Petal Length, and Petal Width.
- **3 species**: _Iris-setosa_, _Iris-versicolor_, _Iris-virginica_.

Data is loaded from  kaggle an SQLite database:  
`/kaggle/input/iris/database.sqlite`

---

## **Project Workflow**
1. **Import Libraries**  
   - `pandas`, `matplotlib.pyplot`, `sqlite3`, `seaborn`, `numpy`
   - `LogisticRegression`, `DecisionTreeClassifier`, `train_test_split`, `plot_tree`

2. **Load and Explore Data**
   - Retrieve the dataset using SQL queries.
   - Display sample records (`df.head()`) and check data types (`df.info()`).
   - Visualize distribution (`sns.scatterplot()`, `value_counts().plot(kind="bar")`).

3. **Data Preprocessing**
   - Extract **features (X)** and **target variable (y)**.
   - Split dataset into **training (90%)** and **testing (10%)** sets.

4. **Baseline Accuracy Calculation**
   - The baseline accuracy is determined by the highest frequency species in the dataset.

5. **Model Training**
   - **Logistic Regression** (`LogisticRegression(max_iter=1000)`)
   - **Decision Tree Classifier** (`DecisionTreeClassifier(random_state=42)`)

6. **Evaluation Metrics**
   - Training and validation accuracy are measured using `.score()`.
   - Feature importance is analyzed and visualized.

7. **Decision Tree Visualization**
   - Using `plot_tree()`, the trained decision tree is displayed.

---

## **Results**
- **Logistic Regression Accuracy**:
  - **Training Accuracy:** `0.9777`
  - **Validation Accuracy:** `1.0`

- **Decision Tree Accuracy**:
  - **Training Accuracy:** `1.0`
  - **Validation Accuracy:** `1.0`

- **Feature Importance Analysis**:
  - Logistic Regression importance plotted using `np.exp(model_lr.coef_[0])`
  - Decision Tree importance analyzed using `.feature_importances_`

---

## **Visualizations**
- **Scatter plots** showing species separation by petal and sepal measurements.
- **Bar plots** highlighting class distribution.
- **Decision Tree Diagram** displaying model structure.
- **Feature Importance Charts** ranking influential predictors.

---



## **Conclusion**
Both models successfully classified **Iris species** with high accuracy.  
- **Decision Tree** had **100% accuracy**,
- **Logistic Regression** performed well while maintaining interpretability.

