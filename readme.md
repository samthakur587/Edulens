![Edulance](https://github.com/bhaswata08/Edulance/assets/106006087/33db5a20-3ebd-46c6-b5d5-b48f860798d3)
<p align="center">
    <h1 align="center">VECTRA_PIPELINE</h1>
</p>
<p align="center">
    <em>Process video insights, extract essential data.</em>
</p>
<p align="center">
	<!-- Shields.io badges not used with skill icons. --><p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<a href="https://skillicons.dev">
		<img src="https://skillicons.dev/icons?i=fastapi,html,md,py,redis">
	</a></p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [ Overview](#-overview)
- [ Features](#-features)
- [ Repository Structure](#-repository-structure)
- [ Modules](#-modules)
- [ Getting Started](#-getting-started)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Tests](#-tests)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)
</details>
<hr>

##  Overview

The Vectra pipeline is a video processing system built with FastAPI and Celery that utilizes various machine learning models to analyze and extract insights from videos. It includes modules for figure detection, OCR text recognition, speech-to-text conversion, and secure user authentication. The pipeline uses Google OAuth for user sessions and interacts with external APIs like Vectara for language processing and calendar event fetching, creating a personalized learning experience for users. The architecture is distributed, allowing for efficient parallelization of tasks in the video processing workflow.

---

##  Features



---

##  Repository Structure

```sh
└── vectra_pipeline/
    ├── Dockerfile
    ├── __pycache__
    │   ├── pipeline.cpython-310.pyc
    │   └── utils.cpython-310.pyc
    ├── data.json
    ├── models
    │   ├── .gitkeep
    │   ├── figure_detection_approach2
    │   ├── slideclassifier_resnet
    │   └── train_yolov9_figuredetection_customdataset1.ipynb
    ├── ocr.py
    ├── pipeline.py
    ├── readme.md
    ├── requirements.txt
    ├── utils.py
    ├── video_procees.py
    ├── video_processor
    │   ├── .gitignore
    │   ├── __pycache__
    │   ├── app
    │   ├── celerybeat-schedule.bak
    │   ├── celerybeat-schedule.dat
    │   ├── celerybeat-schedule.dir
    │   ├── data.json
    │   ├── load_path.py
    │   ├── mixed_data
    │   ├── readme.md
    │   ├── setup.sh
    │   ├── test_vectra.py
    │   ├── vectara_connect
    │   └── worker
    └── yolo.py
```

---

##  Modules

<details closed><summary>.</summary>

| File                                                                                             | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---                                                                                              | ---                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| [Dockerfile](https://github.com/samthakur587/vectra_pipeline/blob/master/Dockerfile)             | Builds Docker image for the video processing pipeline using Python 3.12.3. Installs required system dependencies, copies project files, and installs Python packages from requirements.txt files. Sets working directory to video_processor and exposes FastAPI and Redis ports. Commands to start Redis server, Celery worker, and run the FastAPI application.                                                                         |
| [data.json](https://github.com/samthakur587/vectra_pipeline/blob/master/data.json)               | In this repository, the data.json file serves as a configuration store for pipeline parameters. It is essential for the correct functioning of various components, including OCR and figure detection models, in the vectra_pipeline system.                                                                                                                                                                                             |
| [ocr.py](https://github.com/samthakur587/vectra_pipeline/blob/master/ocr.py)                     | Extracts text from image files using the EasyOCR library for an extensive dataset of PDF documents. Iterates through data files and processes each image, saving extracted text into separate text files. Enhances readability by handling various authors and titles.                                                                                                                                                                   |
| [pipeline.py](https://github.com/samthakur587/vectra_pipeline/blob/master/pipeline.py)           | The pipeline.py file triggers the execution of two scripts, video_procees.py and ocr.py, which process videos using an unspecified method and extract data from frames with Easy OCR.                                                                                                                                                                                                                                                    |
| [requirements.txt](https://github.com/samthakur587/vectra_pipeline/blob/master/requirements.txt) | In this repository, the requirements.txt file lists essential libraries for project execution. These include easyOCR, efficientnet_pytorch, numpy, pandas, and others, necessary for handling image and text processing tasks, using deep learning models, and managing videos.                                                                                                                                                          |
| [utils.py](https://github.com/samthakur587/vectra_pipeline/blob/master/utils.py)                 | Downloads videos from user-provided URLs and saves them to the designated directory in the repository. Extracts video metadata and creates necessary folders before storing files. If the video title is not already in the data.json file, it adds the new metadata.                                                                                                                                                                    |
| [video_procees.py](https://github.com/samthakur587/vectra_pipeline/blob/master/video_procees.py) | In this Python script, videos are processed and converted into images, audio, and text. The `download_video` function retrieves metadata while saving the video to disk. `video_to_images` creates a sequence of frames from the video, while `video_to_audio` extracts the audio data. `audio_to_text` converts speech in the audio file into text. These processed outputs are saved and handled further within the parent repository. |
| [yolo.py](https://github.com/samthakur587/vectra_pipeline/blob/master/yolo.py)                   | Explore a Python script named yolo.py within the vectra\_pipeline repository. This script performs object detection using the pre-trained YOLov8n model, obtained from Roboflow, on an input image. It then annotates the image with bounding boxes and labels using the Supervision library. The annotated image is finally displayed.                                                                                                  |

</details>

<details closed><summary>models</summary>

| File                                                                                                                                                                      | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---                                                                                                                                                                       | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| [train_yolov9_figuredetection_customdataset1.ipynb](https://github.com/samthakur587/vectra_pipeline/blob/master/models/train_yolov9_figuredetection_customdataset1.ipynb) | The `vectra_pipeli` repository is designed for building an extensive data pipeline system. This specific code file, located at a presumed path within the repository, is essential for handling the transformation phase of our data processing pipelines (assuming its a transformer function).The transformer function in this code is responsible for converting raw or ingested data into a more usable, structured format for downstream processing. Critically, it adds value by performing specific data manipulations, extracting key features, and cleaning up irrelevant or redundant data points to improve overall pipeline performance.As part of the larger data pipeline architecture within `vectra_pipeli`, this transformer function interacts with both data ingestion (loader) modules and downstream processing (analyzer/model) components. This enables it to receive raw or unstructured input data, process and refine that data, and finally deliver more structured output suitable for advanced analytical and machine learning algorithms.By leveraging the power of this transformer function within `vectra_pipeli`, we achieve significant gains in our overall data processing efficiency and flexibility, while reducing complexity in our downstream pipelines by providing them with clean, structured input data. |

</details>

<details closed><summary>models.figure_detection_approach2</summary>

| File                                                                                                                                     | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---                                                                                                                                      | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| [figure_detection.py](https://github.com/samthakur587/vectra_pipeline/blob/master/models/figure_detection_approach2/figure_detection.py) | Detect figures from image files using TensorFlows Object Detection API with given parameters. Perform text detection (using frozen East model) to check for figure validity based on minimum text entropy threshold. Sort the output figures according to their positions and save them to specified directories. Adjust settings for optimal results.                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| [rlsa.py](https://github.com/samthakur587/vectra_pipeline/blob/master/models/figure_detection_approach2/rlsa.py)                         | Transform binary images by applying Run Length Smoothing horizontally and vertically using given value for consecutive pixel positions. This method, called rlsa(), helps extract Region-of-interest from documents in the vectra_pipeline project.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| [ssa.py](https://github.com/samthakur587/vectra_pipeline/blob/master/models/figure_detection_approach2/ssa.py)                           | The `models/figure_detection_approach2/ssa.py` file in the given repository is responsible for implementing a particular figure detection approach using Singular Value Decomposition (SVD). The main purpose of this module is to preprocess and extract features from input data that will be later utilized by machine learning models for detecting figures in various contexts. This code adheres to the approach number two mentioned in other files, hence the name approach2. The SVD technique, specifically, is employed for image processing tasks aimed at reducing image dimensions while preserving most of its information. Consequently, these extracted features contribute significantly to enhancing detection performance in downstream machine learning pipelines. |
| [text_detection.py](https://github.com/samthakur587/vectra_pipeline/blob/master/models/figure_detection_approach2/text_detection.py)     | Detect text within an image using OpenCVs built-in Extended Multi-Scale Sliding Windows (EMSSD) algorithm for text detection. Apply non-maxima suppression to suppress weak, overlapping bounding boxes. Output scaled coordinates of detected texts.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

</details>

<details closed><summary>models.slideclassifier_resnet</summary>

| File                                                                                                                                                     | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---                                                                                                                                                      | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| [custom_nnmodules.py](https://github.com/samthakur587/vectra_pipeline/blob/master/models/slideclassifier_resnet/custom_nnmodules.py)                     | This file introduces the `AdaptiveConcatPool2d` class for torch deep learning models. It combines the functionality of AdaptiveAvgPool2d and AdaptiveMaxPool2d into one layer, providing flexible pooling operations in the vectra_pipeline repository.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| [inference.py](https://github.com/samthakur587/vectra_pipeline/blob/master/models/slideclassifier_resnet/inference.py)                                   | Load a trained model from specified file path using PyTorch. Provide image input, receive classification result, confidence percentage, and feature extraction if requested. Use efficient transfer learning architectures like ResNet or EfficientNets.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| [slide_classifier.py](https://github.com/samthakur587/vectra_pipeline/blob/master/models/slideclassifier_resnet/slide_classifier.py)                     | Classifies frames from a directory using a pre-trained ResNet model, saving correctly classified frames into designated folders and calculating the percentage of incorrect classifications. This script is part of the Vegtra pipeline, which automates figure detection in scientific presentations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| [slide_classifier_helpers.py](https://github.com/samthakur587/vectra_pipeline/blob/master/models/slideclassifier_resnet/slide_classifier_helpers.py)     | The slide_classifier_helpers.py file in models/slideclassifier_resnet changes ReLU activations to Mish ones for improved neural network performance within the Vectra pipeline repository, resulting in better classification results.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| [slide_classifier_torchmain.py](https://github.com/samthakur587/vectra_pipeline/blob/master/models/slideclassifier_resnet/slide_classifier_torchmain.py) | Importing necessary libraries, including PyTorch and Pytorch Lightning, a library that simplifies creating deep learning models using Python.2. Defining arguments with the help of argparse module for setting various parameters like batch_size, learning rate, etc.3. Utilizing logging to monitor and record important events during model training.4. Instantiating and defining custom classes for network architecture, data processing, and optimization. This is achieved through the imports of `nn` (PyTorch Neural Network API) and custom-defined modules within the repository like `utils`, and `pipeline`.5. Eventually, the code sets up the main entry point to run PyTorch Lightning models in a simplified fashion while supporting various data loaders, which helps train, validate, and test this specific model. |

</details>

<details closed><summary>video_processor</summary>

| File                                                                                                                           | Summary                                                                                                                                                                                                                                                                    |
| ---                                                                                                                            | ---                                                                                                                                                                                                                                                                        |
| [celerybeat-schedule.dir](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/celerybeat-schedule.dir) | In this project, the celerybeat-schedule.dir file at video_processor directory defines Celery Beat scheduling configuration. It sets entry intervals and versions, timezone information, and UTC enabled flag for task scheduling in the pipeline architecture.            |
| [data.json](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/data.json)                             | In this repository, the `data.json` file in the `video_processor` directory serves as a collection of metadata for videos. It contains information such as video titles, authors, and view counts, enabling efficient video identification and access within the pipeline. |
| [load_path.py](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/load_path.py)                       | This script imports and reads data.json file, extracting last entry's information. Subsequently, it identifies base directory for mixed_data, lists its contents, and determines transcript folder by using author's name, then lists its contents as well.                |
| [setup.sh](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/setup.sh)                               | Sets up Python environment with specified version, installs dependencies, installs Redis server, starts Redis and Celery worker, and launches the FastAPI application within the given directory structure.                                                                |
| [test_vectra.py](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/test_vectra.py)                   | In the video_processor directory, `test_vectra.py` serves as the test suite for the Vectra pipeline implementation. It ensures functional correctness and validates the performance of various modules such as figure detection and OCR processing.                        |

</details>

<details closed><summary>video_processor.app</summary>

| File                                                                                                                 | Summary                                                                                                                                                                                                                                                                                         |
| ---                                                                                                                  | ---                                                                                                                                                                                                                                                                                             |
| [auth.py](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/app/auth.py)                   | Initialize FastAPI application with authentication middleware using Google OAuth. The app sets up session storage, serves static files, and mounts template directory. Home page displays login or welcome messages based on user presence; handles Google sign-in, error handling, and logout. |
| [config.py](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/app/config.py)               | Configures Celery broker and result backend as Redis running on localhost, sets client ID and secret for Google OAuth authentication within the video processing app.                                                                                                                           |
| [main.py](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/app/main.py)                   | Handling user authentication using Google OAuth, receiving video URLs as API requests, processing videos through task queue with Celery, and fetching responses from an external Vectra AI chatbot service.                                                                                     |
| [ocr.py](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/app/ocr.py)                     | Extract text from images within specified directories using EasyOCR and save outputs as separate files. This Python script processes images based on information in data.json, performing optical character recognition and saving results in output\_[Author] folders.                         |
| [pipeline.py](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/app/pipeline.py)           | In video_processor/app/pipeline.py, initiate video processing by running utilities and scripts. Call utils.py for URL preprocessing, execute video_process.py for video transformation, and invoke ocr.py to extract data from frames using Easy OCR.                                           |
| [requirements.txt](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/app/requirements.txt) | In the video_processor directory, the requirements.txt file lists essential dependencies for running this component of the Vectra pipeline. Notable packages include easyocr for text recognition, efficientnet_pytorch for deep learning models, and moviepy for video processing.             |
| [tasks.py](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/app/tasks.py)                 | Utils.py` for initial processing and `video_procees.py` for frame extraction, OCR text recognition with `ocr.py`, and data extraction from JSON files. After processing, the transcript is uploaded to vectara through `upload_file` function.                                                  |
| [utils.py](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/app/utils.py)                 | Downloaded and processed given video URL, saving it to the mixed\_data directory along with its frames and audio file. Metadata was also added to the data.json file for future reference if the video title was not already included.                                                          |
| [video_procees.py](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/app/video_procees.py) | In the video_procees.py file, functions are defined to download videos, convert them into sequences of images and save them, extract audio from videos, recognize speech from audio, and save transcripts as text files. The videos metadata is also stored during video downloading.           |

</details>

<details closed><summary>video_processor.app.templates</summary>

| File                                                                                                                   | Summary                                                                                                                                                                                                                                                                                                                                                |
| ---                                                                                                                    | ---                                                                                                                                                                                                                                                                                                                                                    |
| [error.html](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/app/templates/error.html)     | Create error pages for your video processing application. This HTML template file in video_processor/app/templates/ folder handles rendering custom error messages with a given error message as the argument.                                                                                                                                         |
| [home.html](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/app/templates/home.html)       | This HTML template design in video_processor/app/templates/home.html establishes the initial user experience for Edulance.AI by integrating a visually appealing layout featuring login options through both Google and form inputs.                                                                                                                   |
| [login.html](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/app/templates/login.html)     | Initialize Firebase with provided config, set up FirebaseUI for Google sign-in, load Google API Client, authenticate user and fetch calendar events upon successful sign-in.                                                                                                                                                                           |
| [secure.html](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/app/templates/secure.html)   | The video_processor/app/templates/secure.html file is responsible for producing HTML content for secured user sessions, welcoming logged-in users with the message You are Logged In Bro!!!. It enhances the user experience in this pipeline project, contributing to its architecture as a dedicated element of interactive, personalized interface. |
| [terms.html](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/app/templates/terms.html)     | Generate an impactful HTML template for your terms of service in the video processor app within the vectra_pipeline repository.                                                                                                                                                                                                                        |
| [welcome.html](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/app/templates/welcome.html) | Initialize Firebase with provided configuration. Configure Firebase UI authentication with Google sign-in, Terms of Service, and scopes. Start the AuthUI instance upon DOM loading and set onAuthStateChanged listener for user state changes. Utilize gapi library to load API client and start app if user is signed in, or sign out otherwise.     |

</details>

<details closed><summary>video_processor.mixed_data</summary>

| File                                                                                                        | Summary                                                                                                                      |
| ---                                                                                                         | ---                                                                                                                          |
| [test.txt](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/mixed_data/test.txt) | In the video processing pipeline, this test file stores varied input data for testing purposes within the mixed_data folder. |

</details>

<details closed><summary>video_processor.vectara_connect</summary>

| File                                                                                                               | Summary                                                                                                                                                                                                                                                                                                                                      |
| ---                                                                                                                | ---                                                                                                                                                                                                                                                                                                                                          |
| [.env](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/vectara_connect/.env)           | Connects and authenticates the pipeline with the Vectara API. This file contains environment variables, including the Vectara API key and customer ID, facilitating seamless integration of OCR functionality within the video processing pipeline.                                                                                          |
| [chat.py](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/vectara_connect/chat.py)     | Interacting with Vectaras API, this Python script retrieves text summaries from provided queries using Vectaras advanced language processing capabilities. It sets up the API key, customer ID, and corpus ID as environment variables for authentication. Upon query submission, it prints out the top three text summaries in JSON format. |
| [upload.py](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/vectara_connect/upload.py) | The `upload.py` script in `vectara_connect` folder initiates file uploads to the Vectara platform. It retrieves API keys, customer ID, and corpus ID from environment variables then sends files, one at a time, as multipart requests using the `requests` library.                                                                         |

</details>

<details closed><summary>video_processor.worker</summary>

| File                                                                                                                    | Summary                                                                                                                                                                                               |
| ---                                                                                                                     | ---                                                                                                                                                                                                   |
| [celery_worker.py](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/worker/celery_worker.py) | Initiates Celery worker for a data processing pipeline using Redis as both broker and result backend, enabling the distributed execution of tasks in this video processing component.                 |
| [config.py](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/worker/config.py)               | Configures Celery broker and result backend in the video processing worker component of the vectra pipeline repository, utilizing Redis as both the message broker and task results storage system.   |
| [requirements.txt](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/worker/requirements.txt) | In the video_processor repository, the worker requirements file specifies the needed packages for running Celery tasks and handling Redis messages within the video processing pipeline architecture. |

</details>

<details closed><summary>video_processor.worker.mixed_data.author_1</summary>

| File                                                                                                                        | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---                                                                                                                         | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [test.txt](https://github.com/samthakur587/vectra_pipeline/blob/master/video_processor/worker/mixed_data/author_1/test.txt) | The provided file `test.txt` is located in the `mixed_data/author_1` directory within the `worker` subfolder of the `video_processor` package, in the broader `vectra_pipeline` repository.The main purpose of the `vectra_pipeline` project appears to involve processing videos and extracting data or insights from them. The repository contains various components such as models, a pipeline script, utilities, an OCR module, and a video processor module. This organization suggests that this is a machine learning or computer vision application aimed at analyzing video content.The file `test.txt` seems to be a simple text document containing an introspective note on understanding intangible assets, which may have no direct relationship to the primary functionality of the codebase. However, given its presence within the project directory hierarchy, it might possibly serve as auxiliary data for training or testing ML models or providing additional context during development and debugging. |

</details>

---

##  Getting Started

**System Requirements:**

* **Python**: `version x.y.z`

###  Installation

<h4>From <code>source</code></h4>

> 1. Clone the vectra_pipeline repository:
>
> ```console
> $ git clone https://github.com/samthakur587/vectra_pipeline
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd vectra_pipeline
> ```
>
> 3. Install the dependencies:
> ```console
> $ pip install -r requirements.txt
> ```

###  Usage

<h4>From <code>source</code></h4>

> Run vectra_pipeline using the command below:
> ```console
> $ python main.py
> ```

###  Tests

> Run the test suite using the command below:
> ```console
> $ pytest
> ```

---

##  Project Roadmap

- [X] `► INSERT-TASK-1`
- [ ] `► INSERT-TASK-2`
- [ ] `► ...`

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Report Issues](https://github.com/samthakur587/vectra_pipeline/issues)**: Submit bugs found or log feature requests for the `vectra_pipeline` project.
- **[Submit Pull Requests](https://github.com/samthakur587/vectra_pipeline/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/samthakur587/vectra_pipeline/discussions)**: Share your insights, provide feedback, or ask questions.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/samthakur587/vectra_pipeline
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="center">
   <a href="https://github.com{/samthakur587/vectra_pipeline/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=samthakur587/vectra_pipeline">
   </a>
</p>
</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

[**Return**](#-overview)

---
