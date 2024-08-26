# **VersaVizyon**

VersaVizyon is a flexible and powerful Python script designed to predict various target variables from your dataset. With the ability to handle any column as the target variable (`y`), this script utilizes machine learning models, including Decision Trees and Random Forests, and is optimized through hyperparameter tuning for accurate predictions.

## **Features**

- **Flexible Target Variables**: Choose any column in your dataset as the target variable.
- **Model Options**: Utilize Decision Trees and Random Forests for predictions.
- **Hyperparameter Tuning**: Automatically searches for optimal model parameters using `GridSearchCV`.
- **Performance Metrics**: Evaluates models with MAE, MSE, RMSE, and R2 Score.
- **Cross-Validation**: Ensures robust performance through cross-validation.

## **Installation**

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/VersaVizyon.git
   cd VersaVizyon
   ```

2. **Install the required packages**:

   It is recommended to use a virtual environment. First, ensure you have `pip` installed, then:

   ```bash
   pip install -r requirements.txt
   ```

   **Requirements file (`requirements.txt`)**:

   ```plaintext
   pandas
   numpy
   scikit-learn
   ```

## **Usage**

1. **Prepare your dataset**:

   Ensure your dataset is in CSV format and contains the necessary columns.

2. **Update the file path**:

   Open the script and update the `data_file_path` variable to point to your dataset:

   ```python
   data_file_path = "path_to_your_dataset.csv"
   ```

3. **Run the script**:

   Execute the script to train and evaluate models:

   ```bash
   python versa_vizyon.py
   ```

   The script will output the best model parameters and performance metrics.

## **Script Overview**

### **`versa_vizyon.py`**

- **Data Loading**: Reads the dataset from a CSV file.
- **Data Preparation**: Handles missing values and encodes categorical variables.
- **Model Training**: Trains Decision Trees and Random Forests.
- **Hyperparameter Tuning**: Uses `GridSearchCV` to find the best parameters for the Decision Tree model.
- **Evaluation**: Outputs performance metrics including MAE, MSE, RMSE, and R2 Score.

## **Example**

Here's an example of how to use VersaVizyon to predict a specific target variable:

1. **Set the target variable**:

   ```python
   y = dataset["TargetColumn"]
   ```

2. **Run the script** to obtain predictions and model performance metrics.

## **Contributing**

Contributions are welcome! To contribute to VersaVizyon:

1. Fork the repository.
2. Create a feature branch:
   
   ```bash
   git checkout -b feature/new-feature
   ```

3. Commit your changes:
   
   ```bash
   git commit -am 'Add new feature'
   ```

4. Push to the branch:
   
   ```bash
   git push origin feature/new-feature
   ```

5. Create a Pull Request.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Contact**

For any questions or feedback, feel free to reach out:

- Google form: https://forms.gle/9gPZC1m3kmHNNoPf7

---

Thank you for using VersaVizyon! Happy predicting!
```
