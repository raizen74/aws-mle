### AWS Certified Machine Learning Engineer (MLA-C01) EXAM NOTES - David Galera, July 2025

- [Glue](#glue)
- [Glue DataBrew](#glue-databrew)
- [SageMaker](#sagemaker)
- [SageMaker automatic model tuning (AMT)](#sagemaker-automatic-model-tuning-amt)
- [SageMaker Model Registry](#sagemaker-model-registry)
- [SageMaker Experiments](#sagemaker-experiments)
- [SageMaker processing jobs](#sagemaker-processing-jobs)
- [SageMaker AI endpoints](#sagemaker-ai-endpoints)
- [SageMaker file modes](#sagemaker-file-modes)
- [SageMaker Pipelines](#sagemaker-pipelines)
- [SageMaker Model Monitor](#sagemaker-model-monitor)
- [SageMaker Clarify](#sagemaker-clarify)
- [SageMaker debugger](#sagemaker-debugger)
- [SageMaker TensorBoard](#sagemaker-tensorboard)
- [Comprehend](#comprehend)
- [Kendra](#kendra)
- [Model Interpretability](#model-interpretability)
- [QuickSight](#quicksight)
- [Data Formats](#data-formats)
- [Extra](#extra)

## Glue
You can use AWS Glue ETL jobs to define and perform transformations on your data. You can create **Python** or **Scala** scripts to perform the necessary transformations. Then, AWS Glue will manage the underlying infrastructure to run the job.

AWS Glue **does not provide ready-to-use Docker images** for popular ML frameworks such as TensorFlow and PyTorch.

## Glue DataBrew
Visual data preparation tool that can be used to clean and normalize data, handle missing values, and remove outliers efficiently. It provides a user-friendly interface for data preparation tasks without the need for complex coding.

## SageMaker
Importing models:
- Bring custom **R models** to SageMaker: "bring your own container" option, just write the Dockerfile.
- **SageMaker SDK** includes functions that you can use to **onboard existing Python models** that are written in supported frameworks.
- You must register the models in the **SageMaker model registry** before you can import the models into **SageMaker Canvas**

By default, SageMaker training and deployed inference containers are **internet-enabled**. To prevent the training and inference containers from having access to the internet, you must enable **network isolation**.

## SageMaker automatic model tuning (AMT)
Searches for the most suitable version of a model by running training jobs based on the algorithm and objective criteria. You can use a **warm start tuning job to use the results from previous training jobs as a starting point**. SageMaker can use **early stopping** to compare the current objective metric against the median of the running average of the objective metric.
- `IDENTICAL_DATA_AND_ALGORITHM` setting assumes the same input data and training image as the previous tuning jobs
- `TRANSFER_LEARNING` setting can use different input data, hyperparameter ranges, and other hyperparameter tuning job parameters than the parent tuning jobs.

## SageMaker Model Registry
You can use SageMaker Model Registry to create a catalog of models for production, to manage the versions of a model, to associate metadata to the model, and manage approvals and automate model deployment for CICD. **You would not use SageMaker Model Registry for model re-training**.

## SageMaker Experiments
Feature of **SageMaker Studio** that you can use to automatically create ML experiments by using different combinations of data, algorithms, and parameters. Not used to collect new data.

## SageMaker processing jobs
You can run the custom code on data that is uploaded to Amazon S3. SageMaker processing jobs **provide ready-to-use Docker images** for popular ML frameworks and tools.

SageMaker offers **built-in support** for various frameworks including TensorFlow, PyTorch, scikit-learn, XGBoost, and more.

## SageMaker AI endpoints
- Asynchronous: You can receive responses for each request in **near real time** for **up to 60 minutes of processing time**. There is **no idle cost** to operate an asynchronous endpoint. Designed for use cases where requests can be processed in batches and are not time-sensitive.
- Real-time: Receive responses for each request in real time, can process responses only for **up to 60 seconds**. Real-time endpoints have a **continuous cost**, even when idle. Requires provisioning and managing infrastructure to handle traffic spikes
- Serverless: Receive responses for each request in **real time**, processing time up to 60 seconds. Scale independently. **Memory limit 6 GB and max 200 concurrent requests**. You **cannot configure a VPC** for the endpoint in this solution.
- Batch transform: Process large batches of data and suitable for processing jobs that do not require immediate results. Run inference when you do not need a persistent endpoint.

**Data Capture** is a feature of SageMaker endpoints. You can use Data Capture to record data that you can then use for **training, debugging, and monitoring**. Data Capture runs **asynchronously without impacting production traffic**. You can use the data captured to **retrain the model**.
## SageMaker file modes
- **File mode** downloads training data to a local directory in a Docker container.
- **Pipe mode** streams data directly to the training algorithm.
- **Fast file mode** provides the benefits of both file mode and pipe mode. For example, fast file mode gives SageMaker the flexibility to access entire files in the same way as file mode. Additionally, fast file mode provides the better performance of pipe mode.

**Fast file mode** gives the model the ability to begin training before the entire dataset has finished loading. Therefore, fast file mode **decreases the startup time**. As the training progresses, **the entire dataset will load**. Therefore, you must **have enough space** within the storage capacity of the training instance.

## SageMaker Pipelines
SageMaker Pipelines is a **workflow orchestration service** within SageMaker. It allows users to define end-to-end ML workflows, including steps for data preprocessing, model training, and model evaluation, making it an ideal choice for automating the ML process in response to new data. SageMaker Pipelines supports the use of **batch transforms** to run inference of entire datasets.

**Batch transforms are the most cost-effective inference method** for models that are called only on a periodic basis. Can be triggered by **EventBridge**.

## SageMaker Model Monitor
- **Data Drift** detection: `DefaultModelMonitor` class to generate statistics and constraints around the data and to deploy a monitoring mechanism that evaluates whether data drift has occurred.
- **Bias** detection: `ModelBiasMonitor` class to create a bias baseline and deploy a monitoring mechanism that evaluates whether the model bias deviates from the bias baseline.
- **Feature attribution** detection: use the `ModelExplainabilityMonitor` class to generate a **feature attribution baseline** and to deploy a monitoring mechanism that evaluates whether the feature attribution has occurred.
- **Quality drift** detection: `ModelQualityMonitor`

## SageMaker Clarify
Designed to detect bias in datasets and model predictions. It can be used to analyze how changes in data distribution affect the model's predictions

Provides comprehensive tools for analyzing the impact of data distribution changes on model bias and fairness. Assess impact of data shift on model performance.

## SageMaker debugger
Helps in identifying training issues such as vanishing gradients, exploding gradients, and tensor saturation. Relevant for debugging model training processes.

## SageMaker TensorBoard
Capability of SageMaker that you can use to visualize and analyze intermediate tensors during model training. SageMaker with TensorBoard provides full visibility into the model training process, including debugging and model optimization. You can analyze the intermediate activations and gradients during training

## Comprehend
Amazon Comprehend provides the ability to **locate and redact PII entities in English or Spanish text documents**. You can easily process and anonymize personal information.

## Kendra
Search service that uses natural language processing and advanced ML algorithms to return specific answers to search questions from your data.

## Model Interpretability
- Partial dependence plots (PDP): Identify the difference in the predicted outcome as an input feature changes.
- Shapley values: **Feature attribution**. Quantify the contribution of each feature in a prediction.
- Difference in proportions of labels (DPL): DPL is a metric that you can use to **detect pre-training bias**. You can use DPL to avoid ML models that could potentially be biased or discriminatory.

## QuickSight
You can use QuickSight to make predictions for a column in a model dataset.

## Data Formats
- `Apache ORC` (Optimized Row Columnar) is another columnar storage format that is designed for high performance and efficient storage of data. While it can be a good choice for analytical processing, `Apache Parquet` is generally preferred for its wider adoption, better performance, and support for **schema evolution in semi-structured data**.
- `Apache Parquet` is a columnar storage format optimized for analytics **well-suited for storing semi-structured data with schema evolution requirements**. It supports efficient compression, encoding, and is suitable for both batch and interactive analytics on semi-structured data. It also supports schema evolution, which allows modifications to the data structure over time. It provides efficient storage, high performance for batch and interactive analytics, and supports complex nested data structures. This makes it a suitable choice for analytical processing needs.
- `CSV`, not suited for semi-structured data with schema evolution

## Extra
- Lexical search: Compares the words in a search query to the words in documents, matching word for word
- XGBoost `pos_weight`: helps address any bias in the dataset
- ROC-AUC: Useful metric for evaluating model performance across different thresholds
- F1 Score is the harmonic mean of `precision` and `recall`
- Lowering learning rate -> **reduces the variance in the gradients**, which prevents oscillations in the training accuracy
- Dropout: Prevents neurons adapting to much to specific features, forcing the network to learn more robust features -> Stabilizing the training process
- L1: Adding a penalty proportional to the absolute value of the weights, encourages **sparsity**.
- Batch normalization: Stabilize the learning process by normalizing inputs to each layer