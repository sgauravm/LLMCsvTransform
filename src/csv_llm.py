import re
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
import time
import json
from datetime import datetime
import numpy as np



######################################################################################
# Task1. Extracting column name similar to template column name.
# Steps: 
#     1. Divided the problem into two sub problems: 
#         a. extracting table column information and pattern
#         b. Extracting similar column names
#     2. Created two template and two llm chain to extract a. and b. separately. The 
#        value from a. was passed to b.
#     3. Each table (Template table and Target Table), was converted to json format 
#        to be passed as prompt. Only 5 rows were sampled from each table to keep
#        the prompt small.
######################################################################################


# Initialising LangChain OpenAI
llm = OpenAI(temperature=0.01,max_tokens=512)
# llm = OpenAI(openai_api_key="<Your API Key>",temperature=0.01,max_tokens=512)

# Prompt template for extracting column data information such as data type and data format.
col_info_template_text = '''Consider  data in table_a in json format

## Table Data

table_a in json format:
{table_data_json}

## Instruction
Give the data type and data format  of each column for table_a

## Answer
'''

# Prompt template for selecting similar column name to template column name
col_select_template_text = '''Consider  data in table_a and table_b in json format

## Table Information

table_a data in json format:

{template_table_json}

table_b data in json format:

{target_table_json}

table_a column data type and data format:

{template_table_format}

table_b column data type and data format:

{target_table_format}


## Instruction
1. Identify the column from table_b that corresponds to the {col_name} column in table_a based on the data type, data format and column name. The data type of the answer should be similar to data type of {col_name} column.
2. In case there are other similar columns in table_b that are similar to {col_name} column of table_a, in termas of data_type and data format, provide them as candidate columns. Provide this value only in case where there is ambiguity in main answer.
3. Give reasoning behind the answer of 1 and 2.

## Answer format
The answer should be in the following format:
Answer: <column name from table_b. Based on instruction 1.>
Candidate columns: <comma separated column name from table_b. To be provided only in case of ambiguity based on instruction 3. If no candidate is there, leave it blank>
Reason: <Reason for choosing the answer and candidate columns. Based on instruction 3.>

## Answer
'''
class ParseColNameInfo(BaseOutputParser):
    """Parse the output string of LLM to convert into json format. This class will be passed to LLM chain to parse LLM output.
    """    
    def parse(self,txt):
        """Parse the output of the LLM for processing column type.

        Args:
            txt (str): Output of LLM.

        Returns:
            dict: Structured format from the input unstructured text.
        """        
        result = {}
        
        # Searching for pattern in model output to extract similar column value
        pattern = r"Answer: (.*?)(?:\n|$)"
        matches = re.findall(pattern, txt, re.DOTALL)
        if matches:
            result["col_name"] = matches[0].strip()
        else:
            result["answer"]="No answer found."
        
        # Searching for pattern in model output to extract other similar columns in case of ambiguity.
        pattern = r"\nCandidate columns: (.*?)(?:\n|$)"
        matches = re.findall(pattern, txt, re.DOTALL)
        if matches:
            result["similar_col_name"] = matches[0].strip()
        else:
            result["similar_col_name"]="No answer found."
        
        # Searching for pattern in model output to extract reason for the above answer.
        pattern = r"\nReason:\s*([^#]*)"
        matches = re.findall(pattern, txt, re.DOTALL)
        if matches:
            result["reason"] = matches[0].strip()
        else:
            result["reason"]="No answer found."
        result['gen_text']=txt
        return result

# Initializing the prompt for extracting column information.
col_info_template = PromptTemplate(
    input_variables = ["table_data_json"],
    template = col_info_template_text
)

# Initializing the prompt for selecting similar column
col_select_template = PromptTemplate(
    input_variables = ["template_table_json","target_table_json","template_table_format","target_table_format","col_name"],
    template = col_select_template_text
)

##################REMOVE SLEEP####################################
def get_col_map(target,template):
    """Extracts relevant columns from target table which are similar to template table.
    In addition it also extracts other similar columns as well as reason for arriving at the particular answer.

    Args:
        target (pandas.DataFrame): The target dataframe that needs to be transformed.
        template (pandas.DataFrame): The template dataframe to which the target has to be transformed.

    Returns:
        dict: Dictionary containing similar column, other similar column and the reason why a particular answer was choosen.
    """ 
    #Initializing LLMChain for extracting column info   
    col_info_chain = LLMChain(llm=llm, prompt=col_info_template)
    
    #Initializing LLMChain for extracting relevant column
    col_select_chain = LLMChain(llm=llm, prompt=col_select_template,output_parser=ParseColNameInfo())
    
    # For storing final structured result of LLM
    col_map = {}
    
    # Converting 5 rows of tables to json format to be fed into prompt
    template_table = template.dropna().sample(min(5,len(template))).to_json()
    target_table = target.sample(min(5,len(target))).to_json()
    
    # Calling LLM chain to get column format of tables to be passed to column selection LLM chain
    template_table_format = col_info_chain.run({"table_data_json":template_table})
   
    target_table_format = col_info_chain.run({"table_data_json":target_table})
    
    # Looping over each column of teplate table to find similar column in target table
    col_names = template.columns.tolist()
    for col_name in col_names:
        input_vars = {
            "template_table_json":template_table,
            "target_table_json" : target_table,
            "col_name" : col_name,
            "template_table_format":template_table_format.strip(),
            "target_table_format":target_table_format.strip()
        }
        # Calling LLM chain to get similar colum to template column
        res = col_select_chain.run(input_vars)
        col_map[col_name]=res
    return col_map


######################################################################################
# Task2. Generating python function to transform the selected target column to the
#        format of template column.
# Steps: 
#     1. The task was a nuanced task and also the llm will have to make decision on 
#        whether to do transformation or not.
#     2. In order to tackle this complication took few shot training approach by 
#        giving model 4 exapmles. 
#     3. Finally the code was extracted and an additional line was added to handle 
#        null values if any in the data.
######################################################################################

#Few shot learning template to generate tranformation code
trans_template_text = '''## Code Instruction:

Consider two list of values:
list A:['10-05-2023', '07-05-2023', '04-05-2023', '09-05-2023', '05-05-2023']
List B:['05/08/2023', '05/05/2023', '05/04/2023', '05/03/2023', '05/10/2023']

1. Write a python code to convert values of List B to the format of the values of list A. The function should be of format "def transform(val):"
2.If the data format of both lists are same, then do not do any transformation and return the following code:
def transform(val):
	return val
3. Imports permitted:( import re , from datetime import datetime). Assume these libraries are already included and do not include in the answer.

## Code:
def transform(val):
  date_object = datetime.strptime(val, '%m/%d/%Y')
  return date_object.strftime('%d-%m-%Y')
  
## Code Instruction:

Consider two list of values:
list A:['Phone', 'Car', 'Car', 'Monitor', 'Monitor']
List B:['Monitor', 'Car', 'Phone, 'Telivision', 'Monitor']

1. Write a python code to convert values of List B to the format of the values of list A. The function should be of format "def transform(val):"
2.If the data format of both lists are same, then do not do any transformation and return the following code:
def transform(val):
  return val
3. Imports permitted:( import re , from datetime import datetime). Assume these libraries are already included and do not include in the answer.

## Code:
def transform(val):
  return val


## Code Instruction:

Consider two list of values:
list A:['GAU999', 'MON721', 'CB825', 'QP924', 'XL543']
List B:['MON-101', 'TRR-171', 'LKK-31', 'STU-181', 'MNT-151']

1. Write a python function to convert values of List B to the format of the values of list A. The function should be of format "def transform(val):"
2.If the data format of both lists are same, then do not do any transformation and return the following code:
def transform(val):
  return val
3. Imports permitted:( import re , from datetime import datetime). Assume these libraries are already included and do not include in the answer.

## Code:
def transform(val):
  val_list = val.split('-')
  if len(val_list) == 2:
    return val_list[0] + val_list[1]
  else:
    return val
    
## Code Instruction:

Consider two list of values:
list A:['John Doe', 'Bob Wilson', 'Michael Brown', 'Frank Jackson', 'Eva Thomas']
List B:['Carol Martinez', 'Frank Jackson', 'Michael Brown', 'Jane Smith', 'David Anderson']

1. Write a python code to convert values of List B to the format of the values of list A. The function should be of format "def transform(val):"
2.If the data format of both lists are same, then do not do any transformation and return the following code:
def transform(val):
  return val
3. Imports permitted:( import re , from datetime import datetime). Assume these libraries are already included and do not include in the answer.

## Code:
def transform(val):
  return val
    
    
## Code Instruction:

Consider two list of values:
list A:{temp_list}
List B:{tar_list}

1. Write a python function to convert values of List B to the format of the values of list A. The function should be of format "def transform(val):"
2.If the data format of both lists are same, then do not do any transformation and return the following code:
def transform(val):
  return val
3. Imports permitted:( import re , from datetime import datetime). Assume these libraries are already included and do not include in the answer.

## Code:
'''

# Initializing prompt template
trans_template = PromptTemplate(
    input_variables = ["temp_list","tar_list"],
    template = trans_template_text
)


def get_trans_func(temp,tar,col_map):
    """The function to get python function for transforming selected target table column to the format of template table column.

    Args:
        temp (pandas.DataFrame): Template dataframe.
        tar (pandas.DataFrame): The target dataframe.
        col_map (dict): Mapping of each column of template table to its corresponding column in target table.

    Returns:
        dict: Dictionary containing all template column name and their corresponding target column and its transformation function.
    """  
    #Initializing LLM chain for genearating transformation function  
    chain = LLMChain(llm=llm, prompt=trans_template)
    
    #For storing final result
    res = {}
    
    # Iterating over each template column and its corresponding target column to generate transformation function.
    for temp_col,tar_col in col_map.items():
        # Getting list of 5 values for template and target column
        temp_list = temp[temp_col].sample(min(len(temp),5)).tolist()
        tar_list = tar[tar_col].sample(min(len(tar),5)).tolist()
        
        # Initializing variable list to get prompt from prompt template
        trans_var_list = {
                "temp_list":temp_list,
                "tar_list": tar_list
            }
        
        # Calling LLM chain to get transformation function
        func_str = chain.run(trans_var_list)
        
        # Extra code is added to handle null value in data
        res[temp_col]={"col_name":tar_col,"func_str":func_str.strip().replace("def transform(val):",'''def transform(val):\n  if val is None or val is np.nan:\n    return val''')}

    return res

def string_to_function(func_str):
    """This converts a python function string to executable function.

    Args:
        func_str (str): Python function string.
    Returns:
        function: Executable python function.
    """    
    def wrapper(*args, **kwargs):
        exec(func_str, globals(), locals())
        function_name = func_str.strip().split('(')[0].split(' ')[-1]
        return locals()[function_name](*args, **kwargs)
    
    return wrapper
