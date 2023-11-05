 # **GLPA (Graphic loop painter)**
- [**GLPA (Graphic loop painter)**](#glpa-graphic-loop-painter)
  - [Summary](#summary)
  - [Description](#description)
  - [Coding convention](#coding-convention)
    - [1. Objective](#1-objective)
    - [2. Naming Rules](#2-naming-rules)
    - [3. Coding style](#3-coding-style)

## Summary
Source code for GLPA (Graphic loop painter) library.Personal work, started for the purpose of using in 3D games.
The creator is a Japanese person who is not fluent in English, so DeepL is used for English text as appropriate. Therefore, there is a possibility that mistranslations may exist. If you find one, please contact the creator.

## Description


## Coding convention
### 1. Objective
This library is a personal work, but here are some things to keep in mind to make it more readable and easier to modify for distribution or for your own review

### 2. Naming Rules
| Naming Rule Name         | Use                |
|---------------|-------------------|
| P
ascalCase    | Class name              |
| camelCase     | Function name, Variable name           |
| snake_case    | File name, Header file argument name |
| SNAKE_CASE    | Macro name, Structure name         |
| tagSNAKE_CASE | Structure tag name            |

### 3. Coding style
The following coding style was set up for personal production. Therefore, detailed explanations may be omitted in some cases. Please read with this in mind.
- Use doxygen comments for clarity.

- The maximum number of horizontal characters on a page is 120. If you use vscode, it is recommended to add the following code to setting.json for easier coding.     
    ```json
    "editor.rulers": [
        120
    ],
    ```
- If the number of horizontal characters exceeds 120, change the way the () is written so that it does not exceed 120. An example of how to change this is shown below.

    ```C++
    if
    (
        /* Writing Arguments
        *
        *
        */ 
        
    )
    {
        // Describe the process
    }
    ```

- **Never make ad hoc statements. Make sure that the information is easy to understand for you and others to see later.**

