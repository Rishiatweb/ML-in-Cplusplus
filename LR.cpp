#include <iostream>
#include <cmath>   // For std::sqrt, std::pow
#include <numeric> // No longer needed for Eigen's mean, sum, etc.
#include <algorithm> // No longer needed for Eigen's array operations


#include <Eigen/Dense>

// Using namespace Eigen to simplify Eigen object declarations
using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * @class LinearRegression
 * @brief Implements a simple linear regression model using gradient descent.
 *
 * This class provides functionality to train a linear regression model
 * to predict a target variable (y) based on a single feature (x).
 * It uses Eigen library for all matrix/vector operations and data storage,
 * completely avoiding std::vector.
 * It includes data normalization, cost function calculation (Mean Squared Error),
 * and parameter optimization using gradient descent.
 */
class LinearRegression {
public:
    /**
     * @brief Constructor for LinearRegression.
     * Initializes the model parameters (theta) to zeros.
     * The model expects two parameters: theta_0 (bias) and theta_1 (weight for the feature).
     */
    LinearRegression() : theta(2) {
        theta.setZero(); // Initialize theta_0 and theta_1 to 0.0
        mean_X = 0.0;
        std_X = 1.0; // Default to 1.0 to prevent division by zero if std_dev is 0
    }

    /**
     * @brief Trains the linear regression model using gradient descent.
     * @param X_train_raw The training feature vector (N x 1, where N is the number of samples).
     * This expects the raw feature values, without a bias column initially.
     * @param y_train The training target vector (N x 1).
     * @param learning_rate The step size for gradient descent (alpha).
     * @param num_iterations The number of iterations to run gradient descent.
     */
    void train(const VectorXd& X_train_raw, const VectorXd& y_train,
               double learning_rate, int num_iterations) {
        m = X_train_raw.size(); // Number of training examples

        // Perform data normalization on the input feature (X_train_raw)
        // Store mean and std dev for later prediction
        mean_X = X_train_raw.mean(); // Eigen's mean() method
        std_X = calculate_std_dev(X_train_raw); // Custom std_dev for consistency

        // Handle case where standard deviation is zero (e.g., all X values are the same)
        if (std_X == 0.0) {
            std_X = 1.0; // Prevent division by zero; effectively no normalization
        }

        // Normalize the feature vector
        VectorXd X_normalized = (X_train_raw.array() - mean_X) / std_X;

        // Construct the design matrix X for training.
        // It will have two columns: a column of ones for the bias (theta_0)
        // and the normalized feature column for theta_1.
        MatrixXd X_design(m, 2);
        X_design.col(0).setOnes();       // First column for bias (x_0 = 1)
        X_design.col(1) = X_normalized; // Second column for the feature (normalized x_1)

        // Gradient Descent Loop
        for (int iter = 0; iter < num_iterations; ++iter) {
            // Calculate predictions using current theta values
            // h_theta(x) = X * theta
            VectorXd predictions = X_design * theta;

            // Calculate errors (difference between predictions and actual values)
            // errors = predictions - y_train
            VectorXd errors = predictions - y_train;

            // Calculate the gradient for each parameter
            // Gradient = (1/m) * X_transpose * errors
            VectorXd gradients = X_design.transpose() * errors / m;

            // Update the parameters (theta)
            // theta = theta - learning_rate * gradients
            theta -= learning_rate * gradients;

            // Optional: Print the cost function value periodically to monitor convergence
            if (iter % (num_iterations / 10 > 0 ? num_iterations / 10 : 1) == 0) {
                double current_cost = calculate_cost(X_design, y_train);
                std::cout << "Iteration " << iter << ", Cost: " << current_cost << std::endl;
            }
        }
        std::cout << "Training complete." << std::endl;
        std::cout << "Learned Parameters (theta_0, theta_1): " << theta.transpose() << std::endl;
    }

    /**
     * @brief Calculates the Mean Squared Error (MSE) cost function.
     * J(theta) = (1 / (2*m)) * sum((h_theta(x) - y)^2)
     * @param X_design The design matrix (with bias column).
     * @param y The actual target vector.
     * @return The calculated cost value.
     */
    double calculate_cost(const MatrixXd& X_design, const VectorXd& y) const {
        VectorXd predictions = X_design * theta;
        VectorXd errors = predictions - y;
        // Sum of squared errors, divided by 2m
        return (errors.array().square().sum()) / (2.0 * m);
    }

    /**
     * @brief Predicts the target value for a new, unnormalized input feature.
     * The input feature will be normalized internally using the mean and std dev
     * learned during training.
     * @param X_unnormalized The raw (unnormalized) input feature value.
     * @return The predicted target value.
     */
    double predict(double X_unnormalized) const {
        // Normalize the new input feature using the stored mean and standard deviation
        double X_normalized = (X_unnormalized - mean_X) / std_X;

        // Construct the feature vector for prediction: [1, normalized_X]
        VectorXd X_predict_vec(2);
        X_predict_vec(0) = 1.0;         // Bias term
        X_predict_vec(1) = X_normalized; // Normalized feature

        // Calculate prediction: h_theta(x) = theta_0 * 1 + theta_1 * X_normalized
        // This is equivalent to X_predict_vec.transpose() * theta
        return X_predict_vec.dot(theta);
    }

private:
    VectorXd theta;      ///< Model parameters (theta_0, theta_1)
    double mean_X;       ///< Mean of the training feature (for normalization)
    double std_X;        ///< Standard deviation of the training feature (for normalization)
    int m;               ///< Number of training examples

    /**
     * @brief Calculates the standard deviation of a vector.
     * Using the formula for population standard deviation: sqrt(sum((x_i - mean)^2) / N).
     * @param vec The input vector.
     * @return The standard deviation.
     */
    double calculate_std_dev(const VectorXd& vec) const {
        double current_mean = vec.mean();
        double sq_diff_sum = (vec.array() - current_mean).square().sum();
        return std::sqrt(sq_diff_sum / vec.size());
    }
};

/**
 * @brief Main function to demonstrate Linear Regression.
 */
int main() {
    // 1. Define your dataset (House Area in sqft, Price in INR Lakhs)
    // Directly use Eigen::VectorXd for data.
    VectorXd X_train_data(5); // Area
    X_train_data << 1000, 1200, 1500, 1800, 2000;

    VectorXd y_train_data(5); // Price
    y_train_data << 50, 58, 70, 82, 90;

    std::cout << "--- Starting Linear Regression Training ---" << std::endl;

    // 2. Create a LinearRegression model instance
    LinearRegression model;

    // 3. Set hyperparameters for training
    double learning_rate = 0.01; // Alpha
    int num_iterations = 1000;   // Number of gradient descent steps

    // 4. Train the model
    model.train(X_train_data, y_train_data, learning_rate, num_iterations);

    std::cout << "\n--- Making Predictions ---" << std::endl;

    // 5. Make predictions on new, unseen data
    double new_area1 = 1700; // New house area in sqft
    double predicted_price1 = model.predict(new_area1);
    std::cout << "Predicted price for " << new_area1 << " sqft: " << predicted_price1 << " INR Lakhs" << std::endl;

    double new_area2 = 2500;
    double predicted_price2 = model.predict(new_area2);
    std::cout << "Predicted price for " << new_area2 << " sqft: " << predicted_price2 << " INR Lakhs" << std::endl;

    double new_area3 = 900;
    double predicted_price3 = model.predict(new_area3);
    std::cout << "Predicted price for " << new_area3 << " sqft: " << predicted_price3 << " INR Lakhs" << std::endl;

    return 0;
}
