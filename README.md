  # Film Shot Type Classification
<p align="center">
  <img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/134f8c92-58a7-40d2-a8d8-9c89ff11d1a2" />
</p>
  
  ### Project goal
  Who doesn't love going to the movies? So what if machines could watch movies too. The goal of this project was to predict a simple, measurable property of film frames by shot type and correlate that to narrative structure. By classifying shots as close-up, medium, or wide, I aimed to quantify film directors' visual storytelling patterns.

  ### Data
  - 100 training data frames were generated randomly from film-grab.com by clicking "Random Post" without repeats.
  - 21 testing data frames were generated selectively by clicking "Random Post" to vary equally across the three classes.
  - Each frame was labeled based on shot type according to my personal judgement (as a film enthusiast) of visual framing.

### Features
3 simple, measuarable features were extracted from each frame:
- Edge density: the approximate amount of visual detail in each frame
- Face fraction: the proportion of the frame occupied by a face
- Average brightness: mean pixel intensity

These features were used as inputs for a multiclass logistic regression model.

### Model
- Algorithm: Multiclass logistic regression
- Training: Gradient descent optimization
- Regularization: L2 regularization
- Objective: Predict shot type based on visual features

Implementation of multiclass logistic regression was taken from this article: https://medium.com/data-science/multiclass-logistic-regression-from-scratch-9cc0007da372

### Results
(Taken from standard output) \
Training accuracy: 0.62 \
Test accuracy: 0.47619047619047616 \
Test loss: 1.1195895127842932 \
Final training loss: 0.9827020861557935 \
Generalization gap = 0.1368874266

### Reflection
Training loss decreases over epochs = 100000. Test accuracy is noticeably lower than training accuracy, but for an initial attempt, the model learns from the features as Test accuracy > 1/3, the random accuracy for 3 classes. This shows us that the selected features do capture a trend related to shot type, and it is informative, but limited, but there is overfitting occurring. Adjustments could include changing the penalty term in regularization, perhaps increasing the size of the dataset, or scaling the features to be more accurate. After all, in testing, our face detector sometimes fails to recognize faces, our edge density uses arbitrary thresholds, and from personal observation, average brightness is not a consistent indicator of shot type.

### Future directions of project
Machine learning still struggles with understanding narrative structure of long-form multimodal media—for example, film! Studying shot types was my first foray into this realm, and for film directors, they are very useful visual storytelling tools! Wide or long shots often establish setting, distance between characters or environments, and a sense of grandness or scale. Close-ups usually heighten feelings of intimacy, emotion, and vulnerability. Medium shots are the standard in between. By playing with these different distances to the camera, directors incite certain feelings and understanding in the viewer. I plan to keep working on exploratory projects similar to this, combining my interest in film and machine learning and hopefully adding more computer vision, multimodal, and LLM components!

<p align="center">
  <img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/6b01f89b-df21-4855-9c47-59520b1aa7d9" />
</p>

### Dependencies
- Python 3.x
- NumPy
- SciPy
- OpenCV
