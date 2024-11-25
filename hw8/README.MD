# Assignment: Multilayer Perceptron (MLP) for Handwritten Digit Recognition

## Dataset Description:
- **Training dataset**: `optdigits.tra`
- **Testing dataset**: `optdigits.tes`

## Data Format:
- Each row in the training and testing datasets represents a handwritten digit image.
- **Columns 0 to 63**: Input features composed of 8x8 pixel images' grayscale values, reflecting the image's grayscale levels.
- **Column 64 (the 65th column)**: Output class (label), ranging from 0 to 9, corresponding to different handwritten digits.

## Assignment Tasks:
1. Construct a Multilayer Perceptron (MLP) model using **Numpy**.
2. Train your MLP model using the training dataset (`optdigits.tra`).
3. Evaluate your MLP model's performance using the testing dataset (`optdigits.tes`).
4. Calculate and report the classification accuracy of your model on the test dataset.

## Submission Requirements:
- Implementation code for the MLP model.
- Code for training the model.
- Code for testing and calculating accuracy.
- Report containing the classification accuracy.
