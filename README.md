<p align="center">
    <img src="https://github.com/MiestoMeska/NLP_ToxicCommentsClassification/blob/main/assets/img/Mobbs_Dean-Online-Toxicity.jpg" alt="Toxic" width="75%">
</p>

# Natural Language Processing
# Toxic comment challenge

## Introduction

Online platforms often face challenges in maintaining a respectful environment due to the presence of toxic comments, which can harm user experience and damage a platform's reputation. Manual moderation of these comments is not feasible at scale, making automated solutions necessary.

In this project, we will develop a multi-label classifier to detect various forms of toxicity in forum posts. Using the Kaggle Toxic Comment Classification Challenge dataset, our goal is to build a model that can accurately flag toxic content across six categories, aiding content moderators in maintaining a safer online space.

## Project Task

**Main Goal:** Develop a multi-label classifier to automatically detect and categorize toxic comments into six predefined categories using the Kaggle Toxic Comment Classification Challenge dataset. The model should accurately identify various forms of toxicity, supporting automated content moderation efforts on online platforms.

### Concepts to Explore

In this project, we build on our deep learning knowledge to tackle a complex classification problem using advanced techniques and tools. Our focus will include:

- **Multi-Label Classification:** We will construct a model that can assign multiple labels to each input, identifying various types of toxic behavior in comments. This is essential for recognizing the complexity and overlapping nature of toxic content.

- **Binary Cross-Entropy Loss:** Since this is a multi-label classification problem, we'll use Binary Cross-Entropy Loss, which is suitable for handling independent binary labels across multiple categories, ensuring the model can learn to distinguish each type of toxicity effectively.

- **Transfer Learning with DistilBERT:** Utilizing the DistilBERT tokenizer and model, we'll apply a pre-trained transformer model fine-tuned for our specific task. This approach leverages the power of a state-of-the-art language model, enabling us to handle the nuances of natural language in toxic comments.

- **Custom Model Architecture:** We’ll implement a classifier on top of DistilBERT that includes a ReLU activation and Dropout for regularization. The architecture is designed to refine the representations from DistilBERT and adapt them to our multi-label classification task. Additionally, a multihead model was created during the process, where each head focuses on a specific label, allowing the model to perform binary classification for each toxicity label independently. This approach enables more flexibility and targeted learning for each class, which is especially useful in handling the varying distributions of toxicity categories within the dataset.

- **Optimization Techniques:** The model will be trained using the AdamW optimizer, known for its efficiency and adaptive learning rate capabilities. Additionally, we'll employ a learning rate scheduler (StepLR) to systematically reduce the learning rate, helping the model converge more effectively.



### Project Content

#### Data

- **Acquisition:** The dataset used for this project is provided by the [Kaggle Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). This dataset contains user comments from Wikipedia, labeled for various types of toxicity, including toxic, severe toxic, obscene, threat, insult, and identity hate.
   
- **Exploration:**  [An Exploratory Data Analysis (EDA)](https://github.com/MiestoMeska/NLP_ToxicCommentsClassification/blob/master/notebooks/1.EDA.ipynb) provides insights into the distribution of toxic and non-toxic comments, the balance of different toxicity classes, and the characteristics of the dataset.

- **Augmentation:** [The Data augmentation notebook](https://github.com/MiestoMeska/NLP_ToxicCommentsClassification/blob/master/notebooks/2.data_augmentation.ipynb) contains the process of tackling the imbalance of the classes in the given dataset. 

#### Model Creation



- **Baseline Model:** For this project, a **DistilBERT-based model** was selected. DistilBERT is a smaller, faster, and lighter version of BERT, retaining most of its language understanding capabilities while being more computationally efficient. This makes it well-suited for the task of multi-label classification in detecting toxic comments.
The process of training the model is provided in [Base Model Training Notebook.](https://github.com/MiestoMeska/NLP_ToxicCommentsClassification/blob/master/notebooks/3.train_base_model.ipynb).

- **Routine for the Training of the models:** In this project, a standard approach to model training was adopted, but multiple variations were explored to fine-tune performance. [Model Training Routine Notebook](https://github.com/MiestoMeska/NLP_ToxicCommentsClassification/blob/master/notebooks/4.train_model_routine.ipynb) contains the used routine for training the models.
 Various models were trained using different gradient accumulation values, which allowed for the simulation of larger batch sizes and improved gradient updates. Beyond just experimenting with training parameters, multiple architectures were evaluated, including multihead models and multihead models with shared layers. The multihead models enabled distinct classification heads for each label, while the shared-layer models explored combining representations to improve learning across classes with shared characteristics. This iterative routine led to the training of multiple models, and after evaluation, a few of the most successful ones were selected for further analysis. Since the primary goal of this project was multi-label classification, the model with the highest multi-label F1 score was chosen for final evaluation.

![Multilabel_f1_acc](https://github.com/MiestoMeska/NLP_ToxicCommentsClassification/blob/main/assets/img/models_comp_metric_val_multilabel_acc.JPG)

![Multilabel_f1_score](https://github.com/MiestoMeska/NLP_ToxicCommentsClassification/blob/main/assets/img/models_comp_metric_val_multilabel_f1.JPG)

The two graphs above represent the validation accuracy and F1 score for the multi-label classification task across different model versions during the training process.

Given the goal of maximizing performance in multi-label classification, Multihead Model Version 4 was selected for final evaluation as it demonstrated the highest overall accuracy and F1 score across the validation dataset, consistently outperforming other models in the project scope.

- **Chosen Model Evaluation:** This section of the project involves fine-tuning the classification thresholds for each label to optimize the model's precision and recall. [The Selected Model Evaluation Notebook](https://github.com/MiestoMeska/NLP_ToxicCommentsClassification/blob/master/notebooks/5.selected_model_eval.ipynb) covers the process of determining the optimal thresholds using by testing the model on unseen test data to evaluate its performance. It also includes detailed steps on calculating performance metrics such as accuracy, F1 score, and precision-recall for each label. The notebook documents the entire evaluation process, ensuring that the model generalizes well to new data and meets the project's classification goals.

### Conclusions of the Project

In this project, we successfully developed a multi-label classifier capable of detecting various forms of toxic behavior in online comments. By leveraging the DistilBERT model with multihead classification layers, we were able to build an architecture that effectively identifies toxic content across six predefined categories. Through extensive experimentation with different model architectures and training strategies, we identified the most effective model by evaluating its performance based on validation metrics such as accuracy and F1 score.

**Suggestions for Future Improvements**

-**More Thorough Data Augmentation:** In addition to the back-translation techniques used to balance the dataset, more sophisticated augmentation strategies such as synonym replacement could be applied. This would generate a wider variety of training examples, helping the model generalize better, particularly for the underrepresented classes. Synonym replacement could introduce subtle variations in phrasing that mimic the diversity of toxic language in real-world data.

-**Experimenting with Different Model Bases:** While DistilBERT was chosen for its efficiency and relatively strong performance, experimenting with larger, more complex models like BERT or even RoBERTa could provide a performance boost.

-**Refining Training Routines:** Currently, the base of the model is frozen, meaning the DistilBERT layers are not being fine-tuned during training. Unfreezing these layers and training the full model could lead to better representations for the task, allowing the model to adapt more closely to the specific nuances of the toxic comment classification dataset. Careful experimentation with partial unfreezing or gradual unfreezing could be explored, ensuring that the language model doesn't forget its pre-trained general language understanding while adapting to the task.

-**Additional Regularization Techniques:** Further exploration of regularization techniques like weight decay, layer-wise learning rate decay, or label smoothing could help prevent overfitting, particularly in the presence of imbalanced data or noisy labels.

By exploring these avenues, we can potentially increase the robustness and accuracy of the model, making it even more effective in real-world content moderation scenarios.

