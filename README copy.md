# FinBOS: Financial Video Summarization

## Overview

Financial videos contain a vast amount of valuable information, ranging from market analysis to investment strategies. However, extracting insights from lengthy videos can be time-consuming. FinBOS aims to address this issue by automatically summarizing financial videos, allowing users to grasp key points without watching the entire content.

## Installation

To run FinBOS locally, follow these steps:

*Note: Create a Python environment with Python version 3.10.11 and install torch with CUDA.*

1. Clone the repository:

    ```bash
    git clone https://github.com/mskbasha/FinBOS.git
    cd FinBOS
    git clone https://github.com/haofeixu/gmflow.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```


3. Ensure you have access to a GPU with at least 40GB of memory if you intend to process more than one video simultaneously.

## Usage

1. Open the `run.ipynb` notebook in the repository.

2. Follow the instructions provided in the notebook to load the pre-trained models and resources.

3. Prepare your financial video file(s) or provide a URL to the video(s) you want to summarize.

4. Execute the cells in the notebook to begin the summarization process.

5. Once the summarization process is complete, review the generated summaries and extracted keywords.
