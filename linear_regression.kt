import java.util.Random

fun main() {
    val seed = 42L
    val random = Random(seed)

    // Generate dataset
    val X = (1..100).map { 2 * random.nextDouble() }
    val y = X.map { 4 + 3 * it + random.nextGaussian() } // Using Gaussian for noise

    // Split into training and test sets
    val trainSize = (X.size * 0.8).toInt()
    val X_train = X.take(trainSize)
    val y_train = y.take(trainSize)
    val X_test = X.drop(trainSize)
    val y_test = y.drop(trainSize)

    // Print dataset sizes
    println("trainSize: $trainSize")
    println("X_train.size: ${X_train.size}; X_train: $X_train")
    println("X_test.size: ${X_test.size}; X_test: $X_test")
    println("y_train.size: ${y_train.size}; y_train: $y_train")
    println("y_test.size: ${y_test.size}; y_test: $y_test")

    // Add bias term to X_train and X_test
    val X_train_b = X_train.map { listOf(1.0, it) }
    val X_test_b = X_test.map { listOf(1.0, it) }

    // Initialize weights
    var theta = listOf(random.nextGaussian(), random.nextGaussian())

    // Learning rate
    val learningRate = 0.01
    val nIterations = 1000
    val m = X_train.size

    // Gradient Descent
    for (iteration in 0 until nIterations) {
        val yPred = X_train_b.map { it[0] * theta[0] + it[1] * theta[1] }
        val errors = y_train.zip(yPred) { yi, ypi -> yi - ypi }

        val gradients = List(2) { j ->
            -2.0 / m * errors.zip(X_train_b) { ei, xi -> ei * xi[j] }.sum()
        }

        theta = theta.zip(gradients) { t, g -> t - learningRate * g }
    }

    // Make predictions on the test set
    val y_trainPred = X_train_b.map { it[0] * theta[0] + it[1] * theta[1] }
    val yTestPred = X_test_b.map { it[0] * theta[0] + it[1] * theta[1] }

    // Calculate Mean Squared Error (MSE)
    val mse = y_test.zip(yTestPred) { yi, ypi -> (yi - ypi).pow(2) }.average()

    println("Theta: $theta")
    println("Mean Squared Error (MSE): $mse")


}

// Extension function for power
fun Double.pow(exp: Int): Double {
    return Math.pow(this, exp.toDouble())
}
