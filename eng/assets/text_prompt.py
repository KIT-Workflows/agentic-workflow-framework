
import yaml, os, textwrap, re

from ..interfaces.conda_path import get_conda_sh_path


current_path = os.path.dirname(os.path.abspath(__file__))
current_path = os.path.dirname(current_path)
#yml_path = os.path.join(current_path, 'SET_ME_UP.yaml')
#
#with open(yml_path) as file:
#    settings = yaml.load(file, Loader=yaml.FullLoader)
#
#
#MAIN_DIR = settings['MAIN_DIR']
#QE_ENV = settings['QE_ENV']
#conda_path = get_conda_sh_path()
#if conda_path:
#    CONDA_PATH = conda_path.as_posix()
#    ACTIVATE_COMMAND = f"source {CONDA_PATH} && conda activate {QE_ENV} && cd {MAIN_DIR} && echo $PWD"
#else:
#   ACTIVATE_COMMAND = f"cd {MAIN_DIR} && echo $PWD"
#   print(f"conda path not found, the ACTIVATE_COMMAND is {ACTIVATE_COMMAND}")
#
#QE_COMMAND_ = "export OMP_NUM_THREADS=1 && mpirun -n 8"

QE_GEN_PATTERN = r"```fortran\n(.*?)```"
default_prompt_answer_template = """**Explain Quantum Espresso Technical Documents**

Explain the main points and details related to the following question, based on the provided technical documents from Quantum Espresso.

**Documents:**

{documents}

**User Question:**

{question}

**Task:** Present a clear and concise explanation of the key concepts, algorithms, and methodologies relevant to the user's question, as supported by the provided technical documents. Break down complex technical information into easily understandable language, highlighting important details and implications but avoid giving an example.

**Output:** A detailed and informative response that:

1. Clearly states the main points related to the user's question
2. Provides an explanation of the relevant concepts and algorithms from the Quantum Espresso documents
3. Highlights important details and implications relevant to the question
4. Uses technical vocabulary and notation from the documents to support the explanation
5. Is written in a clear and concise manner, avoiding ambiguity and technical jargon whenever possible
6. Write in markdown format with only a header and body.
Please respond in a structured format, using headings and bullet points to facilitate easy understanding."""


markdown_text = """
<div style="font-family: 'Courier New', Courier, monospace;
            background-color: #FAF3E0;
            font-size: 10px;
            color: #800020; 
            padding: 3px;">
{text}
</div>
"""

markdown_table = """<div style="font-family: 'Courier New', Courier, monospace;
background-color: #FAF3E0;
font-size: 10px;
color: #800020; 
padding: 3px;
display: flex;
justify-content: center;
align-items: center;">

{text}
</div>"""


prompt_answer_instruct_ ="""Write a prompt to use with large language models
that takes a list of technical documents from quantum espresso 
(a program from performing first principle calculation in materials science) 
and a question from the user and clearly explains the main points and details, 
based on these given information.

The prompt should generate a detailed answer to the user's question, according to these instructions:
    
    {instructions}

Create a separate section as input and use {{documents}} and {{question}} as place holder
for the documents and the question, which will be added to the prompt, exactly like below:
    
    **Input**
    * **Documents:*** {{documents}}
    * **Question:*** {{question}}

Output prompt:\n"""

prompt_answer = textwrap.dedent(prompt_answer_instruct_)
prompt_answer = re.sub(r' {2,}', ' ', prompt_answer)
prompt_answer = re.sub(r'\n{2,}', '\n', prompt_answer)
prompt_answer_instruct = re.sub(r'^\s+', '', prompt_answer, flags=re.MULTILINE)


X__ = """You are an AI assistant specializing in quantum chemistry calculations using Quantum ESPRESSO.
Create a Quantum Espresso input file for '{calc_type}' calculation of '{chem_formula}' compound.

<CONTEXT>
{tot}
</CONTEXT>

<GENERAL_DOCUMENTATION>
{doc}
</GENERAL_DOCUMENTATION>

<DETAILS>
{features}
</DETAILS>

<RULES>
- From general documentation, write a specific documentation based on the CONTEXT and the DETAILS
- Write the input file for the given calculation type
- Set the value of the variables based on the CONTEXT and the DETAILS
- Use only relevant parameters based on the CONTEXT, DETAILS, and the specific documentation
- Each parameter should be under correct namelist
- In &CONTROL should have 'pseudo_dir': {pseudo_dir} and 'outdir':'{out_dir}' defined, according to the given CONTEXT
- In &CONTROL, use etot_conv_thr = 1.0d-8 and forc_conv_thr = 1.0d-6 as default values, unless specified otherwise
- In &SYSTEM, nat (number of atoms) and ntyp (number of types of atoms) should be defined correctly. ntyp is the number of unique species given in ATOMIC_SPECIES
- &CONTROL, &SYSTEM, &ELECTRONS, must be present and based on CONTEXT and DETAILS
</RULES>

<OUTPUT>
- After writing the specific documentation, create the input file
- Put the code blocks in ```fortran and ```.
- Explain each parameter and the reason for using it in the context
- Review the code to see how the generated code followed the RULES, CONTEXT, and DETAILS, then give some explanation.
</OUTPUT>
"""


X__m = """You are an AI assistant specializing in quantum chemistry calculations using Quantum ESPRESSO.
The USER created a Quantum Espresso input file:
<INPUT_FILE>
{input_file}
</INPUT_FILE>

However, the USER received the following error message after running the Quantum Espresso calculations:

<ERROR_MESSAGE>
{error_msg} 
</ERROR_MESSAGE>

The USER provided the following documentation for modifying the Quantum Espresso input file:
<DOCUMENTATION>
{doc}
</DOCUMENTATION>

<RULES>
- Not all documentation paths are relevant to the query.
- Select the most relevant documentation to the error message.
- You are allowed to remove or add a parameter if necessary.
- The namelists &CONTROL, &SYSTEM, &ELECTRONS, must be in this order exactly
- Do not changed or modified these namelists: ibrav, K_POINTS, CELL_PARAMETERS, ATOMIC_SPECIES, ATOMIC_POSITIONS.
</RULES>

<TASK>
- Explain the error message and the corrections that are necessary to fix the error message, based on INPUT_FILE and documentation and ERROR_MESSAGE 
- Is there any unnecessary variable in the input file that is not requested or needed or contradicting with existing parameter? If yes, remove it.
- Make sure that each variable is in the correct namelist, and the values are correct.
- Apply the corrections that are necessary to fix the error message, by adding, removing, or displacing the variables, as necessary.
- Review the code to see how the generated code followed the given instructions.
</TASK>

<OUTPUT>
- Put the fixed code blocks in ```fortran and ```.
</OUTPUT>
"""
planner_prompts = """You are a helpful assistance in physics and chemistry. 

<User Input>
- Project topic:
{project_topic}
- Information provided by user:
{info_provided}
</User Input>

<CONTEXT>
- The user is working on a project in physics or chemistry.
- The user needs help with the project.
- The logistics of the project such as computational resources are not a concern.
- The user has provided some information about the project.
- The user is expecting help in organization and planning of the project.
- The user is familiar with basics of physics and chemistry.
</CONTEXT>

The user asked about a certain project topic and provided some information and context. 
Then you will do the following:

<STEPS>
1. Explain the user's project topic.
2. Organize the information that user has already provided.
3. Create a list of information that is needed to complete the project.
4. Create a list of steps that the user needs to take to complete the project.
4.1 For each step, create sub-steps that the user needs to take.
5. for each step and sub-step, provide a brief explanation.
</STEPS>

<OUTPUT-1>
- Explanation of the project topic.
- List of information provided by user.
- List of information needed to complete the project.
- List of steps needed to complete the project.
- For each step, list of sub-steps.
- For each step and sub-step, brief explanation.
</OUTPUT-1>

<OUTPUT-2>
- After finishing the output-1, create a mermaid diagram that shows the steps and sub-steps.
- The diagram should be easy to understand and follow.
- Use the name of tools for each step and sub-step, where applicable.
- There is alway a first step and last step in the diagram.
- The output of each step is the input of the next step.
- Create stylish diagrams.
</OUTPUT-2>
"""

prompt_haves_wants_ = """Based on this context:
<CONTEXT>
{context}
</CONTEXT>

Explain what input information should I provide, and how can i use this to get the following information:
<DESIRED_OUTPUT>
{output}
</DESIRED_OUTPUT>

<RULES>
- not all the provided paths are relevant to the query
- some paths are more general and convenient to use. pick that and continue
</RULES>

<OUTPUT>
- concisely explain the context
- pick and explain the relevant paths that should be used
- concisely explain the input information that should be provided
- enumerate what output can i expect 
- concisely explain how to use this information to get the desired output
- do not provide examples
</OUTPUT>
"""

prompt_haves_wants_single = """Based on this context:
<CONTEXT>
{context}
</CONTEXT>

Explain what input information should I provide, and how can i use this to get the following information:
<DESIRED_OUTPUT>
{output}
</DESIRED_OUTPUT>

<OUTPUT>
- concisely explain the context
- concisely explain the input information that should be provided
- enumerate what output can i expect 
- concisely explain how to use this information to get the desired output
- do not provide examples
</OUTPUT>
"""


prompt_funct_create_ = """Create proper inputs_param and fields for this function:
<FUNCTION>
import requests
from typing import List, Dict, Union

{fetch_materials_data_code}
</FUNCTION>

<HAVES>
{haves}
</HAVES>

<WANTS>
{wants}
</WANTS>

based on these instructions:
<INSTRUCTIONS>
{instructions}
</INSTRUCTIONS>

Now make input_param, and fields based on this

<OUTPUT>
- create input_param and output_fields for the function.
- the input_param should be the input parameters of the function, based on the haves 
- the output_fields should be the output fields of the function, based on the wants
- write the input_param and fields and the python code between ```python and ``` to create the function.
</OUTPUT>
"""


match_prompt = """In order to achieve the following goal, we executed a tool and obtained the following output.
<GOAL>
{goal}
</GOAL>

<OUTPUT>
{output}
</OUTPUT>

this output should be used in combination with the goal, to create the input parameters for the next tool in python environment.
<NEXT_TOOL>
{next_tool}
</NEXT_TOOL>

<OUTPUT>
- explain the goal and the output
- explain the next tool
- create the input parameters for the next tool, using the output and the goal in form of a dictionary of arguments and values. the dictionary will be used as (**input_{{next_tool_name}}) in the next tool
- the name of the dictionary should be `input_{{next_tool_name}}`
- do not include the parameters that have set default values
- put the script for that dictionary between ```python and ```
- do not include examples
</OUTPUT>
"""

qe_input_prompt_ = """You are an AI assistant specializing in quantum chemistry calculations using Quantum ESPRESSO.
Create a Quantum Espresso input file for '{calc_type}' calculation of '{chem_formula}' compound, based on the given CONTEXT and DETAILS.

<CONTEXT>
&CONTROL title = '{title}'
{extras}
{tot}
</CONTEXT>

<DETAILS>
{features}
</DETAILS>

<RULES>
- Write the input file for the given calculation type
- Set the value of the variables based on the CONTEXT and the DETAILS
- Use only relevant parameters based on the CONTEXT, DETAILS
- Each parameter should be under correct namelist
- Each parameter should have a correct value type such as integer, real, character, or logical
- The namelists &CONTROL, &SYSTEM, &ELECTRONS, must be in this order exactly
- The cards ATOMIC_SPECIES, ATOMIC_POSITIONS, K_POINTS, and CELL_PARAMETERS must be present and based on CONTEXT and DETAILS
- In &CONTROL should have 'pseudo_dir': {pseudo_dir} and 'outdir':'{out_dir}' defined, according to the given CONTEXT
</RULES>

<OUTPUT>
- Write the input file
- Put the code blocks in ```fortran and ```.
- Review the code to see how the generated code followed the RULES, CONTEXT, and DETAILS, then give some explanation.
</OUTPUT>
"""


keywords_prompt_ref = """You are an AI assistant specializing in quantum chemistry calculations using Quantum ESPRESSO.
Extract the relevant information, search query, conditions, and parameter names from the following text:

<INPUT>
{text}
</INPUT>

<RULES>
- Do not include the section names in the extracted information
- Avoid information related to the compounds and material names
- The information that you extract will be a query for the search engine
- Information taged as default or given or None should be excluded.
</RULES>

<OUTPUT>
- First explain the content of the input
- Create one python list of abbreviations named 'extracted_abbreviations'. If no abbreviations is found then write extracted_abbreviations = None
- Create another python list named 'extracted_information' and put the extracted information in the list.
- Put the two lists between ```python and ```
</OUTPUT>
"""

keywords_prompt_gem = """You are an AI assistant specializing in quantum chemistry calculations using Quantum ESPRESSO.
Extract the relevant information, search query, conditions, and parameter names from the following text:

<INPUT>
{text}
</INPUT>

<RULES>
- Do not include the section names in the extracted information
- Avoid information related to the compounds and material names
- The information that you extract will be a query for the search engine
- Only extract information that is explicitly stated in the input text.
- Do not include information that is tagged as default, given, or None.
- Do not include ambiguous information or suggestions.
</RULES>

<OUTPUT>
- First explain the content of the input
- Create one python list of abbreviations named 'extracted_abbreviations'. If no abbreviations is found then write extracted_abbreviations = None
- Create another python list named 'extracted_information' and put the extracted information in the list.
- Put the two lists between ```python and ```
</OUTPUT>
"""

keywords_prompt = """You are an AI assistant specializing in quantum chemistry calculations using Quantum ESPRESSO.
Extract the relevant information, search query, conditions, and parameter names from the following text:

<INPUT>
{text}
</INPUT>

<RULES>
- Do not include the section names in the extracted information
- Avoid information related to the compounds and material names
- The information that you extract will be a query for the search engine
- Only extract information that is explicitly stated in the input text
- Do not include information that is tagged as default, given, or None
- Do not include ambiguous information or suggestions
- Do not make assumptions about unspecified parameters or conditions
- Exclude any speculative or implied information
</RULES>

<OUTPUT>
- First explain the content of the input
- Create one python list of abbreviations named 'extracted_abbreviations'. If no abbreviations is found then write extracted_abbreviations = None
- Create another python list named 'extracted_information' and put the extracted information in the list
- Put the two lists between ```python and ```
</OUTPUT>
"""
#class Error_KW(dspy.Signature):
    #        """Write the name of variables mentioned in the input text that caused the error."""
    #        input_text = dspy.InputField( desc = 'The input text containing the keyword names.')
    #        error_line = dspy.OutputField(desc = 'The line that contains the error')
    #        variable = dspy.OutputField(desc = 'The list of variables that caused the error')
    #        explanation = dspy.OutputField(desc = 'The explanation and summarize the error')
    #
    #    ekw = dspy.Predict(Error_KW)
error_keywords_prompt = """You are an AI assistant specializing in quantum chemistry calculations using Quantum ESPRESSO.
In running a Quantum Espresso calculation, the user encountered an error message. 

<ERROR_MESSAGE>
{error_message}
</ERROR_MESSAGE>

Read the error message and do the followings:
<RULES>

- Extract the error message
- Identify the list of variables mentioned in the error message, causing the error
- Summarize the error message and provide an explanation

</RULES>

<OUTPUT>
- Create a python dictionary called 'error_keywords'like this:
    error_keywords = {{'error_message': 'The error message',
                        'variables': ['list of variables'],
                        'explanation': 'The explanation and summarize the error'}}
- Put the dictionary between ```python and ```
</OUTPUT>

Make sure to follow the rules and provide the output in the specified format.
"""

cat_prompt_2 = """You are an AI assistant specializing in quantum chemistry calculations using Quantum ESPRESSO.
Analyze the given calculation description and tag list to extract and categorize relevant information.

<CALCULATION_DESCRIPTION>
{calculation_description}
</CALCULATION_DESCRIPTION>

<TAGS>
{conditions}
</TAGS>

<RULES>
- Extract only information explicitly mentioned in the documentation. Look for specific keywords, parameters, and context clues.
- Tags must be Quantum ESPRESSO specific parameters or keywords. Use the provided list of known tags as a reference.
- Determine tag relevance based on direct applicability to the calculation description. Consider the context and purpose of the calculation.
- Irrelevant tags are those that don't apply to the current calculation type or context.
- Final comments should focus on potential issues, common pitfalls, or important considerations related to the calculation.
- Ensure all extracted information maintains technical accuracy and relevance.

<OUTPUT_FORMAT>
Create a Python dictionary named 'extracted_info' with these keys and descriptions:
   - Calculation_description (str, description of the calculation)
   - Relevant_tags (list[str], relevant tags for the calculation)
   - Irrelevant_tags (list[str], tags that are not relevant for the calculation)
   - Final_comments (str, final comments)

Place the dictionary between ```python and ``` markers. 
</OUTPUT_FORMAT>

Make sure to follow the rules and provide the output in the specified format.
"""


ask_ai_2_= """You are an AI assistant specializing in quantum chemistry calculations using Quantum ESPRESSO.
Your task is to analyze an initial calculation description (ICD) provided by the user, categorize the information, suggest clarifications where needed, and generate an expanded and modified version of the ICD.

Here is the ICD provided by the user:
<initial_calculation_description>
{INPUT}
</initial_calculation_description>
---

Carefully review the ICD provided by the user and follow the rules below to complete your task.

<RULES>

1. Categorize Information: 

Extract information relevant to the following categories:
   Chemical_formula:
      - chemical formula of the compound: such as 'A2B3C4'
   Calculation_types:
      - type of calculation: such as 'scf', 'nscf', 'bands', 'relax', 'vc-relax', 'md', 'vc-md'
   Functional_and_method_types:
      - functional and method types: DFT+U, diagonalization, dispersion correction, hybrid functional, gammaDFT, exact exchange, noncolinear calculations
   Cell_and_material_properties:
      - cell and material properties: such as dimensions, periodicity, metallic, non-metallic, insulating
   Pseudopotential_types:
      - pseudopotential type: details about the pseudopotential
   Magnetism_and_Spin_conditions:
      - magnetism and spin conditions: details about the magnetism and spin conditions
   Isolated_systems_and_boundary_conditions:
      - isolated systems and boundary conditions: such as isolated systems, lau boundary conditions, solvent regions
   k_point_settings:
      - k-point settings: such as automatic k-points, single k-point, uniform k-point grid
   Electric_field_conditions:
      - electric field conditions: such as electric field, kubo terms, system with electric field, lelfield, lfcp, optional electric field, tefield
   Occupation_types:
      - occupation types: such as fixed occupations, gaussian smearing, grand canonical ensemble, linear tetrahedron method
   Database_name:
      - mc3d: For 3D materials, bulk systems, and crystals, or when no specification is provided for the given material
      - mc2d: For 2D materials, layered systems such as graphene, monolayers etc., and slabs

For each category:
   1.1 Record the relevant details if present.
   1.2 If no information is provided, label the category as "Not specified."

2. Identify Ambiguities: 
Highlight any ambiguous or unclear details in the ICD. Specify why the information is unclear or could belong to multiple categories.

3. Suggest Clarifications: 
For each "Not specified" or incomplete category, create targeted questions or prompts to help the user provide the missing information. Be specific.

4. Expand and Modify the ICD: 
Using the extracted information and identified ambiguities, create an expanded version of the ICD that organizes the provided details into their respective categories. 
Include placeholders or default values where appropriate and explicitly mark areas needing user input.

   4.1 Create an expanded version of the ICD that organizes the provided details into their respective categories, as a markdown list.
   4.2 For missing or incomplete details, include placeholders or default values where appropriate and explicitly mark areas needing user input.

</RULES>

<OUTPUT>
- Write your analysis in a clear and structured format for the user. Include the expanded and modified ICD. 
The title of the analysis should be "The Analysis of the Initial Calculation Description." Then include the following sections:
   - Original Description: The ICD provided by the user.
   - Extracted Information: Details extracted for each category.
   - Ambiguities: List of ambiguous or unclear details.
   - Suggestions for Clarification: Specific questions or prompts for missing information.
   - Modified Description: The expanded and modified ICD, clearly labeled and organized, as a markdown list.
Use markdown formatting to present your analysis, as header1 for the title and header2 for each section.

- Create a python dictionary named `analysis_dict` with the following keys:
   'description': str, The original ICD description provided by the user.
   'formula': str, The chemical formula of the compound
   'database': str, One of ['mc2d', 'mc3d'], where:
      - mc3d: For 3D materials, bulk systems, and crystals, or when no specification is provided for the given material
      - mc2d: For 2D materials, layered systems such as graphene, monolayers etc., and slabs
   'analysis': dict, The structured analysis and suggestions, organized as:
      'Extracted_Information': dict, Put the extracted information for each category.
      'Ambiguities': list[str], List the ambiguities found in the ICD.
      'Suggestions_for_Clarification': list[str], List specific suggestions for improving the ICD.
   'modified_description': str, The final expanded and modified ICD.
- Put the python dictionary between ```python ``` tags to ensure proper formatting.
</OUTPUT>

Ensure that your response is detailed, covering all specified categories, and that your suggestions for clarification are clear and actionable."""

ask_ai_distill= """You are an AI assistant specializing in quantum chemistry calculations using Quantum ESPRESSO.
Your task is to analyze an initial calculation description (ICD) provided by the user and extract key information.

Here is the ICD provided by the user:
<initial_calculation_description>
{INPUT}
</initial_calculation_description>

<RULES>
1. Extract the chemical formula if present in the ICD
2. Determine the appropriate database:
   - mc3d: For 3D materials, bulk systems, and crystals, or when no specification is provided
   - mc2d: For 2D materials, layered systems such as graphene, monolayers etc., and slabs

<OUTPUT>
Create a python dictionary named `analysis_dict` with the following keys:
   'description': str, The original ICD description provided by the user
   'formula': str, The chemical formula of the compound (or None if not specified)
   'database': str, One of ['mc2d', 'mc3d']

Put the python dictionary between ```python ``` tags.
</OUTPUT>

Make sure to follow the rules and provide the output in the specified format."""

qe_param_prompt_ = """You are an AI assistant specializing in quantum chemistry calculations using Quantum ESPRESSO.
For the following project and the assigned tags, select the appropriate value for the parameter:
<Project>
{proj}
</Project>

<Conditions>
{conditions}
</Conditions>

<Parameter>
{parameter}
</Parameter>

<RULES>
- The type of calculation is important and they are two types. Singlestep: 'scf', 'nscf', 'bands', and Multistep: 'relax', 'vc-relax', 'md', 'vc-md'.
- The tags must be relevant to the calculation type.
- Do not assume any additional information beyond the provided project and tags.

</RULES>

<Ouput>
- First discuss the relation between the parameter and the project, using most recent research and documentation.
- Determine if the parameter is relevant to the project.
- If the parameter is relevant, select the appropriate value from the list below.
- Create one python dictionary called 'parameter_value' with the parameter's name as the key (str) and the value as the value of the parameter (str).
- If the parameter is not relevant, set the value to None.
- Put the dictionary between ```python and ```.
</Output>

Make sure to follow the rules and provide the output in the specified format.
"""


d_qe_param_prompt_ = """You are an AI assistant specializing in quantum chemistry calculations using Quantum ESPRESSO.
For the following project and the assigned tags, select the appropriate value for the parameter:
<Project>
{proj}
</Project>

<Conditions>
{conditions}
</Conditions>

<Parameter>
{parameter}
</Parameter>

<RULES>
- The type of calculation is important and they are two types. Singlestep: 'scf', 'nscf', 'bands', and Multistep: 'relax', 'vc-relax', 'md', 'vc-md'
- The tags must be relevant to the calculation type
</RULES>

<Ouput>
- First discuss the relation between the parameter and the project.
- Determine if the parameter is relevant to the project.
- If the parameter is relevant, select the appropriate value from the list below.
- Create one python dictionary called 'parameter_value' with the parameter's name as the key and the value as the value.
- If the parameter is not relevant, set the value to None.
- Put the dictionary between ```python and ```.
</Output>
"""