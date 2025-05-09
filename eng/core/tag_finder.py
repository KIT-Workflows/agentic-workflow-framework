from dataclasses import dataclass
from typing import Dict, List, Set, Any, Optional
import pandas as pd
import json
import re
import ast
from concurrent.futures import TimeoutError
from ..assets.text_prompt import cat_prompt_2
from ..interfaces.llms_engine import process_in_batches, a_quick_chat

with open('eng/assets/quantum_conditions.json', 'r') as f:
    quantum_conditions = json.load(f)

@dataclass
class TagFinderResult:
    """
    Result of tag finding operation.
    
    Attributes:
        dataframe: Processed results as pandas DataFrame
        relevant_conditions: List of identified relevant conditions
        errors: List of errors encountered during processing
    """
    items: List[str]
    dataframe: pd.DataFrame
    relevant_conditions: List[str]
    errors: List[Dict[str, Any]]

class TagFinder:
    """Handles extraction and processing of tags from calculation descriptions."""
    
    def __init__(self, 
                 model: str = 'metaviso',
                 batch_size: int = 10,
                 timeout: int = 3600,
                 batch_timeout: int = 900,
                 lm_api: Optional[str] = None,
                 conditions: Dict[str, List[str]] = quantum_conditions,
                 sleep_time: int = 30):
        """
        Initialize TagFinder.
        
        Args:
            model: Model name to use for processing
            batch_size: Size of batches for processing
            timeout: Overall timeout in seconds
            batch_timeout: Timeout per batch in seconds
            lm_api: API key for language model
            conditions: Dictionary of quantum conditions to check
        """
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.batch_timeout = batch_timeout
        self.lm_api = lm_api
        self.conditions = conditions
        self.sleep_time = sleep_time

    async def process_description(self, 
                                calculation_description: str,
                                ) -> TagFinderResult:
        """
        Process calculation description to find relevant tags and conditions.
        
        Args:
            calculation_description: Description to process
            conditions: Dictionary of quantum conditions to check
            
        Returns:
            TagFinderResult containing processed data and any errors
        """
        # Generate prompts
        items = [
            cat_prompt_2.format(
                calculation_description=calculation_description,
                conditions=f'{k}\n' + '\n'.join(f"- {i}" for i in v)
            )
            for k, v in self.conditions.items()
        ]

        try:
            # Process batches
            results = await process_in_batches(
                items=items,
                batch_size=self.batch_size,
                async_func=a_quick_chat,
                func_args={
                    'model': self.model,
                    'lm_api': self.lm_api,
                    'extras': {'temperature': 0.0, 'max_tokens': 5000}
                },
                #timeout=self.timeout,
                batch_timeout=self.batch_timeout,
                sleep_time=self.sleep_time,
            )
            
            return self._process_results(items,results)
            
        except TimeoutError as e:
            return TagFinderResult(
                dataframe=pd.DataFrame(),
                relevant_conditions=[],
                errors=[{"error": "Batch processing timeout", "details": str(e)}]
            )

    def _process_results(self, 
                         items: List[str],
                        results: List[tuple], 
                        ) -> TagFinderResult:
        """Process raw results into structured format."""
        processed_data = []
        rel_cond: Set[str] = set()
        errors = []

        for idx, answer in results:
            try:
                processed = self._extract_data_from_response(answer)
                if processed:
                    item_name = list(self.conditions.items())[idx][0]
                    processed_data.append({"Item": item_name, **processed})
                    
                    # Update relevant conditions
                    tags = processed.get('Relevant_tags', [])
                    if isinstance(tags, list):
                        rel_cond.update(tag for tag in tags if isinstance(tag, str))
                    elif isinstance(tags, str):
                        rel_cond.add(tags)
                else:
                    errors.append({
                        "error": "No valid data extracted",
                        "index": idx,
                        "answer": answer
                    })
            except Exception as e:
                errors.append({
                    "error": str(e),
                    "index": idx,
                    "answer": answer
                })

        df = pd.DataFrame(processed_data)
        if not df.empty:
            df = df[['Item', 'Relevant_tags', 'Irrelevant_tags', 'Final_comments']]

        return TagFinderResult(
            items=items,
            dataframe=df,
            relevant_conditions=list(rel_cond),
            errors=errors
        )

    @staticmethod
    def _extract_data_from_response(response: str) -> Optional[Dict[str, Any]]:
        """Extract structured data from response text."""
        match = re.search(r'```python(.*)```', response, re.DOTALL)
        if match:
            extracted_text = match.group(1)
            if '=' in extracted_text:
                json_str = extracted_text.split('extracted_info = ')[1].strip()
                try:
                    return(ast.literal_eval(json_str))
                except Exception as e:
                    return json.loads(json_str)
        return None 