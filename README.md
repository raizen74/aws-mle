### AWS Certified Machine Learning Engineer (MLA-C01) EXAM NOTES - David Galera, July 2025

- [Glue](#glue)
- [Glue DataBrew](#glue-databrew)
- [Athena](#athena)
- [SageMaker](#sagemaker)
- [SageMaker models](#sagemaker-models)
- [SageMaker DataWrangler](#sagemaker-datawrangler)
- [SageMaker automatic model tuning (AMT)](#sagemaker-automatic-model-tuning-amt)
- [SageMaker Model Registry](#sagemaker-model-registry)
- [SageMaker Experiments](#sagemaker-experiments)
- [SageMaker processing jobs](#sagemaker-processing-jobs)
- [SageMaker AI endpoints](#sagemaker-ai-endpoints)
- [SageMaker file modes](#sagemaker-file-modes)
- [SageMaker Jumpstart](#sagemaker-jumpstart)
- [SageMaker Pipelines](#sagemaker-pipelines)
- [SageMaker Model Monitor](#sagemaker-model-monitor)
- [SageMaker Clarify](#sagemaker-clarify)
- [SageMaker debugger](#sagemaker-debugger)
- [SageMaker TensorBoard](#sagemaker-tensorboard)
- [SageMaker deployments](#sagemaker-deployments)
- [SageMaker Feature store](#sagemaker-feature-store)
- [Amazon Managed Workflows for Apache Airflow](#amazon-managed-workflows-for-apache-airflow)
- [AWS Step functions](#aws-step-functions)
- [LakeFormation](#lakeformation)
- [Comprehend](#comprehend)
- [Kendra](#kendra)
- [Model Interpretability](#model-interpretability)
- [Model training](#model-training)
- [QuickSight](#quicksight)
- [Data Formats](#data-formats)
- [Extra](#extra)

## Glue

You can use AWS Glue ETL jobs to define and perform transformations on your data. You can create **Python** or **Scala** scripts to perform the necessary transformations. Then, AWS Glue will manage the underlying infrastructure to run the job.

AWS Glue **does not provide ready-to-use Docker images** for popular ML frameworks such as TensorFlow and PyTorch.

**AWS Glue Data Quality** is designed to evaluate and enforce data quality rules on datasets processed through AWS Glue ETL jobs.

`AWS Glue crawlers`: can analyze the .csv files in Amazon S3 and automatically infer the schema and structure of the data. This creates a table definition in the AWS Glue Data Catalog, enabling the data to be organized and understood.

!["Glue crawlers"](crawlers.jpg)

## Glue DataBrew

Visual data preparation tool that can be used to clean and normalize data, handle missing values, and remove outliers efficiently. It provides a user-friendly interface for data preparation tasks without the need for complex coding.

!["Glue DataBrew"](databrew.jpg)

## Athena

- Athena can **write the filtered results back to S3 in optimized formats like Parquet or ORC**, which significantly improves query performance and reduces costs.
- Athena requires no infrastructure management, making it the most efficient and low-operational solution for querying S3-based CSV data.

## SageMaker

!["Multi-model"](multimodel.jpg)

Importing models:

- Bring custom **R models** to SageMaker: "bring your own container" option, just write the Dockerfile.
- **SageMaker SDK** includes functions that you can use to **onboard existing Python models** that are written in supported frameworks.
- You must register the models in the **SageMaker model registry** before you can import the models into **SageMaker Canvas**

By default, SageMaker training and deployed inference containers are **internet-enabled**. To prevent the training and inference containers from having access to the internet, you must enable **network isolation**.

**Amazon SageMaker Warm Pools** allow reuse of ML compute infrastructure between consecutive training jobs. This significantly reduces startup times because instances remain warm and do not require new provisioning or configuration. Warm Pools work seamlessly with SageMaker training jobs, helping minimize infrastructure startup overhead while ensuring the infrastructure is reused securely and efficiently. This is ideal for use cases where consecutive training jobs are frequent, as in experimentation workflows.

Bring Your Own Container: `Script mode` enables you to write custom training and inference code while still utilizing common ML framework containers maintained by AWS.

## SageMaker models

- **Random Cut Forest (RCF)**: unsupervised algorithm for detecting anomalous data points within a data set.
- **XGBoost** has built-in techniques for handling class imbalance, such as `scale_pos_weight`.
- **Linear Learner** algorithm can handle classification tasks, it is specifically designed to handle class imbalance by adjusting class weights. However, it may not be as effective in capturing complex patterns in the data as more sophisticated algorithms like XGBoost.

## SageMaker DataWrangler

!["SageMaker data wrangler"](data-wrangler.png)

`balance data` operation
!["SageMaker data wrangler balance"](balance.jpg)

`corrupt image` transform is specifically designed to simulate real-world image imperfections, such as noise, blurriness, or resolution changes, during the preprocessing stage

Suited for cleaning, visualizing, and transforming datasets before training machine learning models, but it lacks the machine learning-powered deduplication features provided by `AWS Glue FindMatches`. It **does not have** built-in capabilities for detecting or grouping **duplicate records**, especially those requiring fuzzy matching.

- Minimal operational effort with a visual, interactive interface.
- Access and select your tabular and image data from various popular sources - such as Amazon Simple Storage Service (Amazon S3), on-premises databases, Amazon Athena, Amazon Redshift, AWS Lake Formation, Snowflake, and Databricks - and over 50 other third-party sources - such as Salesforce, SAP, Facebook Ads, and Google Analytics.
- Write queries for data sources using SQL and import data directly into SageMaker from various file formats, such as CSV, Parquet, JSON, and database tables.
- Provides built-in anomaly detection tools and visual insights for streamlined analysis.
- Easily exports transformed datasets to SageMaker for training.
- Provides a **data quality and insights report** that automatically verifies data quality (such as missing values, duplicate rows, and data types) and helps detect anomalies (such as outliers, class imbalance, and data leakage) in your data.
- Offers over 300 prebuilt PySpark transformations and a natural language interface to prepare tabular, timeseries, text and image data without coding

## SageMaker automatic model tuning (AMT)

Searches for the most suitable version of a model by running training jobs based on the algorithm and objective criteria. You can use a **warm start tuning job to use the results from previous training jobs as a starting point**. SageMaker can use **early stopping** to compare the current objective metric against the median of the running average of the objective metric.

- `IDENTICAL_DATA_AND_ALGORITHM` setting assumes the same input data and training image as the previous tuning jobs
- `TRANSFER_LEARNING` setting can use different input data, hyperparameter ranges, and other hyperparameter tuning job parameters than the parent tuning jobs.

## SageMaker Model Registry

!["SageMaker model registry"](model-registry.jpg)

You can use SageMaker Model Registry to create a catalog of models for production, to manage the versions of a model, to associate metadata to the model, and manage approvals and automate model deployment for CICD. **You would not use SageMaker Model Registry for model re-training**.

- **collections**: allow users to logically organize model groups into high-level categories without disrupting the integrity of the underlying model groups or artifacts. Better model management and discoverability at scale.
- Amazon SageMaker Model Registry also integrates with **AWS Resource Access Manager (AWS RAM)**, making it easier to securely share and discover machine learning (ML) models across your AWS accounts

## SageMaker Experiments

Feature of **SageMaker Studio** that you can use to automatically create ML experiments by using different combinations of data, algorithms, and parameters. Not used to collect new data.

## SageMaker processing jobs

You can run the custom code on data that is uploaded to Amazon S3. SageMaker processing jobs **provide ready-to-use Docker images** for popular ML frameworks and tools.

SageMaker offers **built-in support** for various frameworks including TensorFlow, PyTorch, scikit-learn, XGBoost, and more.

## SageMaker AI endpoints

!["SageMaker endpoints"](endpoints.jpg)

- Asynchronous: You can receive responses for each request in **near real time** for **up to 60 minutes of processing time**. There is **no idle cost** to operate an asynchronous endpoint. Designed for use cases where requests can be processed in batches and are not time-sensitive.
- Real-time: Receive responses for each request in real time, can process responses only for **up to 60 seconds**. Real-time endpoints have a **continuous cost**, even when idle. Requires provisioning and managing infrastructure to handle traffic spikes. You can enable **data capture** and integrate it with Clarify to perform bias detection.
- Serverless: Receive responses for each request in **real time**, processing time up to 60 seconds. Scale independently. **Memory limit 6 GB and max 200 concurrent requests**. You **cannot configure a VPC** for the endpoint in this solution. Ideal for workloads that have idle periods between traffic spikes and can tolerate cold starts.
- Batch transform: Process large batches of data and suitable for processing jobs that do not require immediate results. Run inference when you do not need a persistent endpoint.

**Data Capture** is a feature of SageMaker endpoints. You can use Data Capture to record data that you can then use for **training, debugging, and monitoring**. Data Capture runs **asynchronously without impacting production traffic**. You can use the data captured to **retrain the model**.

## SageMaker file modes

!["File modes"](Inputs.jpg)

- **File mode** downloads training data to a local directory in a Docker container.
- **Pipe mode** streams data directly to the training algorithm.
- **Fast file mode**: **Best suited for workloads with many small files. Only available for S3 data**. Provides the benefits of both file mode and pipe mode. For example, fast file mode gives SageMaker the flexibility to access entire files in the same way as file mode. Additionally, fast file mode provides the better performance of pipe mode.

**Fast file mode** gives the model the ability to begin training before the entire dataset has finished loading. Therefore, fast file mode **decreases the startup time**. As the training progresses, **the entire dataset will load**. Therefore, you must **have enough space** within the storage capacity of the training instance.

## SageMaker Jumpstart

With SageMaker JumpStart, you can evaluate, compare, and select FMs quickly based on pre-defined quality and responsibility metrics to perform tasks like article summarization and image generation. SageMaker JumpStart provides managed infrastructure and tools to accelerate scalable, reliable, and secure model building, training, and deployment of ML models.

You can quickly **deploy a pre-trained model and fine-tune it using your custom dataset**. This approach allows you to leverage existing **NLP models**, reducing both development time and computational resources needed for training from scratch.

## SageMaker Pipelines

!["SageMaker Pipelines"](sagemaker-pipelines.jpg)

Visualization as a DAG - Workflows are represented as a directed acyclic graph (DAG), making it easy to visualize dependencies.

You can use **conditional steps in SageMaker Pipelines** to introduce a manual approval step before proceeding to production deployments.

Callback steps are specifically designed to integrate external processes into the SageMaker pipeline workflow, e.g. by using a callback step, the SageMaker pipeline waits until the AWS Glue jobs complete.

Seamless integration with **SageMaker ML Lineage Tracking** - Automatically tracks lineage information, including input datasets, model artifacts, and inference endpoints, ensuring compliance and auditability.

SageMaker Pipelines is a **workflow orchestration service** within SageMaker. Supports versioning, lineage tracking, and automatic execution of workflows, making it the ideal choice for managing end-to-end ML workflows in AWS. It allows users to define end-to-end ML workflows, including steps for data preprocessing, model training, and model evaluation, making it an ideal choice for automating the ML process in response to new data. SageMaker Pipelines supports the use of **batch transforms** to run inference of entire datasets.

**Batch transforms are the most cost-effective inference method** for models that are called only on a periodic basis. Can be triggered by **EventBridge**.

## SageMaker Model Monitor

- **Data Drift** detection: `DefaultModelMonitor` class to generate statistics and constraints around the data and to deploy a monitoring mechanism that evaluates whether data drift has occurred.
- **Bias** detection: `ModelBiasMonitor` class to create a bias baseline and deploy a monitoring mechanism that evaluates whether the model bias deviates from the bias baseline.
- **Feature attribution** detection: use the `ModelExplainabilityMonitor` class to generate a **feature attribution baseline** and to deploy a monitoring mechanism that evaluates whether the feature attribution has occurred.
- **Quality drift** detection: `ModelQualityMonitor`

## SageMaker Clarify

Designed to detect bias in datasets and model predictions. It can be used to analyze how changes in data distribution affect the model's predictions. You specify input features, such as gender or age, and SageMaker Clarify runs an analysis job to detect potential bias in those features

Provides comprehensive tools for analyzing the impact of data distribution changes on model bias and fairness. Assess impact of data shift on model performance.

## SageMaker debugger

Helps in identifying training issues such as vanishing gradients, exploding gradients, and tensor saturation. Relevant for debugging model training processes.

## SageMaker TensorBoard

Capability of SageMaker that you can use to visualize and analyze intermediate tensors during model training. SageMaker with TensorBoard provides full visibility into the model training process, including debugging and model optimization. You can analyze the intermediate activations and gradients during training

## SageMaker deployments

!["SageMaker deployments"](deployments.jpg)

## SageMaker Feature store

A **feature group** is a logical grouping of features, which is the foundation of the SageMaker Feature Store. It defines the schema of the data, such as feature names, types, and metadata. Creating a feature group is the **first step to structure and organize features**.

## Amazon Managed Workflows for Apache Airflow

It requires significant setup and maintenance, and while it can integrate with AWS services, it does not provide the seamless, built-in integration with SageMaker that SageMaker Pipelines offers.

## AWS Step functions

It is more general-purpose and **lacks some of the ML-specific features**, such as **model lineage tracking** and **hyperparameter tuning**, that are built into SageMaker Pipelines.

## LakeFormation

AWS Lake Formation is specifically designed for aggregating and managing large datasets from various sources, including Amazon S3, databases, and other on-premises or cloud-based sources.

- Large-scale ETL operations
- Automate schema inference from on-premises databases

## Comprehend

Amazon Comprehend provides the ability to **locate and redact PII entities in English or Spanish text documents**. You can easily process and anonymize personal information.

!["Comprehend"](comprehend.jpg)

## Kendra

Search service that uses natural language processing and advanced ML algorithms to return specific answers to search questions from your data.

## Model Interpretability

- Partial dependence plots (PDP): Identify the difference in the predicted outcome as an input feature changes.
- Shapley values: **Feature attribution**. Quantify the contribution of each feature in a prediction.
- Difference in proportions of labels (DPL): DPL is a metric that you can use to **detect pre-training bias**. You can use DPL to avoid ML models that could potentially be biased or discriminatory.

## Model training

!["Overfitting"](overfitting.jpg)
!["Metrics"](metrics.jpg)
!["Metrics2"](metrics2.jpg)

In **bagging**, data scientists improve the accuracy of weak learners by training several of them at once on multiple datasets. In contrast, boosting trains weak learners one after another.

**Stacking** involves training a meta-model on the predictions of several base models. This approach can significantly improve performance because the meta-model learns to leverage the strengths of each base model while compensating for their weaknesses.

**Ensemble**: typically does not capture the complex interactions between models as effectively as stacking.

!["Ensemble"](ensemble.jpg)

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
- Remove duplicates -> `AWS Glue FindMatches`:
  - Minimal coding required - The ML-based approach simplifies the deduplication process.
  - Flexible matching logic - Automatically identifies fuzzy matches and near-duplicates.
  - Scalable and serverless - Works seamlessly with large datasets in Amazon S3.
- High scalable and flexible deployment of model e.g. genAI models: `EKS`
