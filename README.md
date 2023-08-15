# LLM Csv Transform

## Solution Approach

The entire task was divided into two broad tasks:

### Task 1: Extracting Column Names Similar to Template Column Name

**Steps:**
1. Divided the problem into two subproblems: 
    - Extracting table column information and data pattern.
    - Extracting similar column names.
2. Created two templates and two Language Learning Model (LLM) chains to extract the subproblems separately. The output from the first subproblem was passed to the second.
3. Each table (Template table and Target Table) was converted to JSON format to be used as prompts. Only a sample of 5 rows from each table was used to keep the prompts concise.

### Task 2: Generating Python Function to Transform the Selected Target Column

**Steps:**
1. This task involved nuanced decision-making, where the Language Learning Model had to decide whether to perform a transformation or not.
2. To tackle this complexity, a few-shot training approach was used by providing the model with 4 examples.
3. Finally, the code was extracted, and an additional line was added to handle any null values in the data.



## Steps to Run the Streamlit App

**Note:** The OpenAI API I am using is a free version which is almost exhausted, so I am not deploying the web interface on public platform. Here, I am providing steps to run the app on your local system.

1. **Clone the Repository:**

```
git clone https://github.com/sgauravm/LLMCsvTransform.git
```

2. **Create a Virtual Environment:**

```
conda create -n llmcsv python=3.10
conda activate llmcsv
```

3. **Set Up Environment Variable `OPENAI_API_KEY`:**
- For Linux/MacOS:
  ```
  export OPENAI_API_KEY="..."
  ```
- Alternatively, you can go to the file `src/csv_llm.py`, comment out line `29`, and uncomment line `30`. Then, pass your API key directly.

4. **Go to the repo directory and install the required packages:**
- Using pip:
```
pip install -r requirements.txt
```
- Using conda (In case pip does not work)
```
conda install --file requirements.txt
```

5. **Run the Streamlit App:**
```
streamlit run Upload_CSV.py
```



## Discussion on Edge Cases

### 1. Template Column with Multiple Target Columns

In some scenarios, the information present in the template column needs to be mapped to multiple columns in the target table. For instance, consider a situation where the template has a column named `Full_Name`, but the target table stores this information across two columns: `First_Name` and `Last_Name`. Similar situations can occur with various data types such as dates, addresses, and more.

**Solution:**
- To address this issue, it's recommended to treat such cases as separate prompts during the transformation process.
- The code responsible for single column selection should be adapted to recognize composite columns.
- A dedicated prompt can be crafted to generate a function capable of handling multiple input columns and processing them into a unified output column.

### 2. Handling Null Values during Transformation

Another edge case emerges when applying transformation code to a column to align it with the template column's data format. Problems can arise when the data contains null values, potentially causing errors in the transformation process.

**Solution:**
- This case has been handled in the current code.
- Additional code segments addressing null value scenarios have been incorporated into the generated transformation function.

These solutions aim to enhance the robustness and reliability of the transformation process, ensuring smoother data alignment and manipulation.



