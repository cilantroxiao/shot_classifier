  # Film Shot Type Classification

  ### Project goal
  The goal of this project was to predict a simple, measurable property of film frames by shot type and correlate that to narrative structure. By classifying shots as close-up, medium, or wide, I aimed to quantify film directors' visual storytelling patterns.

  ### Data
  - 100 training data frames were generated randomly from film-grab.com by clicking "Random Post" without repeats.
  - 21 testing data frames were generated selectively by clicking "Random Post" to vary equally across the three classes.
  - Each frame was labeled based on shot type according to my personal judgement (as a film enthusiast) of visual framing.

### Features
3 simple, measuarable features were extracted from each frame:
- Edge density: the approximate amount of visual detail in each frame
- Face fraction: the proportion of the frame occupied by a face
- Average brightness: mean pixel intensity\

These features were used as inputs for a multiclass logistic regression model.

### Model
- Algorithm: Multiclass logistic regression
- Training: Gradient descent optimization
- Regularization: L2 regularization
- Objective: Predict shot type based on visual features \

Implementation of multiclass logistic regression was taken from this article: https://medium.com/data-science/multiclass-logistic-regression-from-scratch-9cc0007da372

### Dependencies
- Python 3.x
- NumPy
- SciPy
- OpenCV
