# Key–Value Schema for the Smart Knowledge Graph

## Introduction

The key–value schema defines how input parameters and cards from the QE documentation are represented as nodes within the smart KG. Each entry is encoded as a JSON object with specific keys to capture all relevant metadata, including connectivity and activation criteria. This document details the schema for both namelist parameters and card entries, outlines the structure of the `connections` and `conditions` fields, and includes illustrative JSON examples (see Figures \figref{fig\:param} and \figref{fig\:card}).

## Namelist Parameters

Namelist parameters are represented using the following keys:

* `Namelist`
* `Parameter_Name`
* `Value_Type`
* `Default_Values`
* `Description`
* `Possible_Usage_Conditions`
* `Required/Optional`
* `Usage_Conditions`
* `Parameter_Value_Conditions`
* `Relationships_Conditions_to_Other_Parameters_Cards`
* `Final_comments`

## Card Parameters

Card-type entries use a more extensive set of keys to capture their complex syntax:

* `Card_Name`
* `Namelist`
* `Required/Optional`
* `Card_Options`
* `Default_Option`
* `Description`
* `Card_Use_Conditions`
* `Card_Option_Given_Conditions`
* `Syntax_Given_Option`
* `Item_Description`
* `Item_Conditions`
* `General_Syntax`
* `Relationships_Conditions_to_Other_Parameters_Cards`
* `Possible_Usage_Conditions`
* `Final_comments`

## Connections and Conditions

* **`connections`**: Lists node identifiers that are directly linked to this entry, representing semantic or functional relationships extracted from the documentation.
* **`conditions`**: Specifies the criteria (derived from user prompts) that must be met to activate or suggest this node within the recommendation workflow.

## Manual Validation Notes

After automated schema extraction, all entries underwent manual review. Particular attention was paid to:

* Ensuring completeness of the `connections` graph.
* Verifying accuracy of activation `conditions`.
* Adding any missing links or criteria that the LLM failed to capture.

Further community-driven refinement is encouraged to enhance these fields.

## JSON Examples

Example JSON entries illustrating the schema:

* **Parameter** `nspin`: see Figure \figref{fig\:param}
* **Card** `ATOMIC_SPECIES`: see Figure \figref{fig\:card}
